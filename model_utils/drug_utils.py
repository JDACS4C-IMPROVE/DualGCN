"""
author: @cesarasa
"""

# Main imports
import os
import deepchem as dc
import hickle as hkl
# Importing from libraries
from rdkit import Chem


def smiles_to_molecules(smiles_list : list) -> list:
    """The following function aims to convert SMILES to RDKit molecules. And save all the molecules in a list.
    """
    molecules = []
    for smile in smiles_list:
        molecule = Chem.MolFromSmiles(smile)
        molecules.append(molecule)
    
    return molecules

def rdkit_mols_to_dc_features(mols: list) -> list:
    """
    RDKit mols to DeepChem features

    The rdkit_mols_to_dc_features function takes a list of RDKit molecules and returns a list of DeepChem features. 
    It uses the ConvMolFeaturizer class from the DeepChem package to extract features from the molecules.
    """
    ft = dc.feat.ConvMolFeaturizer()
    feature_list = ft.featurize(mols)
    return feature_list

def save_hickles(feature_list, drug_id_list, save_dir) -> None:
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for ii in range(len(feature_list)):
        ft = feature_list[ii]
        obj = (ft.atom_features, ft.canon_adj_list, ft.deg_list)
        f_name = os.path.join(save_dir, str(drug_id_list[ii]) + ".hkl")
        hkl.dump(obj, f_name)
    
    return None 