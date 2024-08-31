"""
2/6/2023

@Author: Ruibo

The provided code contains several functions that are used to convert PubChem Compound IDs (CIDs) to RDKit molecules, and then to DeepChem features. 
    Finally, it saves the features of the molecules in the form of hickle files.

The main function is meta_file_to_hickle(). 
Usage example:
```
from process_gen_mol_graph import meta_file_to_hickle
meta_file_to_hickle("drug_meta.csv", "hickle_file_dir")
```
Drug meta file mush contain a column "PubChem", which is the PubChem CID, and a column "drug_id", which is matched with the response data. 


Current: blocked at how to get the PubCHEM of drugs. Are they provided in the curated data?
"""

import os
import random
import deepchem as dc
import pandas as pd
import requests
from rdkit import Chem
import time
import hickle
from improve_utils import load_single_drug_response_data_v2

def cids_to_rdkit_molecules(pubchem_cids: list):
    """
    CID list to RDKit mol list.

    The cids_to_rdkit_molecules function takes a list of PubChem CIDs and returns a list of RDKit molecules. 
        It sends a GET request to the PubChem database to retrieve the SDF file of the compound with the specified CID. 
        If the status code of the response is 429 (too many requests), it waits for 0.2 seconds and tries again until it succeeds.
    """
    molecules = []
    for pubchem_cid in pubchem_cids:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/CID/{pubchem_cid}/SDF'
        while True:
            r = requests.get(url)
            if r.status_code == 429:
                time.sleep(0.2)
                continue
            molecule = Chem.MolFromMolBlock(r.text)
            molecules.append(molecule)
            break
    return molecules

def rdkit_mols_to_dc_features(mols: list) -> list:
    """
    RDKit mols to DeepChem features

    The rdkit_mols_to_dc_features function takes a list of RDKit molecules and returns a list of DeepChem features. 
    It uses the ConvMolFeaturizer class from the DeepChem package to extract features from the molecules.
    """
    ft = dc.feat.ConvMolFeaturizer()
    feat_list = ft.featurize(mols)
    return feat_list

def save_hickles(feature_list, drug_id_list, save_dir):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ii in range(len(feature_list)):
        ft = feature_list[ii]
        obj = (ft.atom_features, ft.canon_adj_list, ft.deg_list)
        f_name = os.path.join(save_dir, str(drug_id_list[ii]) + ".hkl")
        hickle.dump(obj, f_name)

def meta_file_to_hickle(f_drug_meta_data="data/drug/1.Drug_listMon Jun 24 09_00_55 2019.csv", save_dir="data_new/drug/drug_graph_feat"):
    """
    The main function meta_file_to_hickle reads a CSV file containing drug metadata that includes PubChem CIDs and drug IDs. 
    It extracts CIDs and drug IDs from the CSV file, converts CIDs to RDKit molecules, 
        and then converts the molecules to DeepChem features using the rdkit_mols_to_dc_features function. 
    Finally, it saves the features in the form of hickle files using the save_hickles function.

    input: path to drug meta file containing the drug ID (in study) and the drug PubChem CID. cols names are PubCHEM and drug_id
    output: save hickle files to the save_dir. 
        Each hickle file is named by the drug ID, as <drug_id>.hkl, and contains feat_mat, adj_list, degree_list.
    """
    drug_meta = pd.read_csv(f_drug_meta_data)
    cids = drug_meta.PubCHEM.values  # Make sure the meta file contains column named "PubCHEM"
    drug_ids = drug_meta.drug_id.values  # Make sure the meta file contains column named "drug_id"
    mols = cids_to_rdkit_molecules(cids)
    feat_list = rdkit_mols_to_dc_features(mols)
    save_hickles(feat_list, drug_ids, save_dir)
    return None
    
def improve_utils_to_hickle(drug_response_df: pd.DataFrame, output_dir):
    """
    Convert the PubChem CIDs in improve_utils returned DFs to DeepChem graphs, and save the graphs as hickle files. 
    Input:
        The DataFrame returned by improve_utils.
    Output:
        No return values. The generated graphs will be saved in the given save_dir.load_single_drug_response_data_v2.
        The drug names will be the IMPROVE chem IDs (CID with "PC_")
    """
    try:
        impr_ids = drug_response_df["improve_chem_id"].unique()
        cid_list = [each_cid.replace("PC_", "") for each_cid in impr_ids]
    except KeyError:
        print('Column "improve_chem_id" not found in the passed response DF. The available columns are ', drug_response_df.columns)
        return None

    mols = cids_to_rdkit_molecules(cid_list)
    feat_list = rdkit_mols_to_dc_features(mols)
    save_hickles(feat_list, impr_ids, os.path.join(output_dir, "drug/drug_graph_feat/"))
    return None


#%% Test code
def test_same_graph():
    import networkx as nx
    # test the generated graph is the same to the provided, using network X
    provided = os.listdir("data/drug/drug_graph_feat")
    cid =  random.choice(provided).strip(".hkl")
    print(cid)
    mols = cids_to_rdkit_molecules([cid])
    feat = rdkit_mols_to_dc_features(mols)[0]
    f0, a0, d0 = feat.atom_features, feat.canon_adj_list, feat.deg_list
    f1, a1, d1 = hickle.load("data/drug/drug_graph_feat/{}.hkl".format(cid))
    assert(d0 == d1)
    nn = len(d0)
    g0 = nx.Graph()
    g1 = nx.Graph()
    g0.add_nodes_from(range(nn))
    g1.add_nodes_from(range(nn))
    nx.set_node_attributes(g0, {node: features for node, features in zip(g0, f0)}, 'features')
    nx.set_node_attributes(g1, {node: features for node, features in zip(g1, f1)}, 'features')

    for node, neighbors in enumerate(a0):
        for neighbor in neighbors:
            g0.add_edge(node, neighbor)
    for node, neighbors in enumerate(a1):
        for neighbor in neighbors:
            g1.add_edge(node, neighbor)

    if nx.is_isomorphic(g0, g1):
        print("The graphs are equivalent.")
    else:
        print("The graphs are not equivalent.")
    
    assert(nx.is_isomorphic(g0, g1))


def test_improve_to_hickle():
    split = 0
    source_data_name = "CCLE"
    y_col_name = "auc"
    x = load_single_drug_response_data_v2(
        source=source_data_name,
        split_file_name=f"{source_data_name}_split_{split}_val.txt",
        y_col_name=y_col_name
        )
    improve_utils_to_hickle(x, output_dir="data_new/drug/test")
    return None    

if __name__ == "__main__":
    # Run test code. 
    test_improve_to_hickle()
    test_same_graph()
    meta_file_to_hickle()

