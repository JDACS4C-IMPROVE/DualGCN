"""
@ author: cesarasa

This script is used to preprocess the data from the curated dataset by ANL. 

The main goal is to emulate the original data from the curated dataset by ANL.

NOTE: 

- We only have 24 chemicals (drugs or compounds) in the curated dataset by ANL, contrasting 
with the +200 chemicals in the original dataset.

- Only 411 cell lines. 

- We dont have the information from 7 genes, so we remove them from the gene list and the 
PPI network. 

(how do we find the PPI data? From String? )
"""

import os
import improve_utils as iu
import pandas as pd
from tqdm import tqdm
from time import time as t

gene_list = pd.read_csv('./data/CCLE/gene_list.txt', header= None).values.astype(str).flatten().tolist()
not_in_dataset = ['SEPT6', 'SEPT5', 'FGFR1OP', 'H3F3A', 'C15orf65', 'SEPT9', 'H3F3B', 'CARS']

for gene in not_in_dataset:
    gene_list.remove(gene)

def processing_data(df_ge, df_cn):
    """
    This function is mainly to get the same columns and index in both dataframes.
    """
    # Having same columns
    ge_columns = df_ge.columns
    cn_columns = df_cn.columns
    common_columns = list(set(ge_columns).intersection(cn_columns))
    df_ge = df_ge[common_columns]
    df_cn = df_cn[common_columns]

    # Having same index
    ge_index = df_ge.index
    cn_index = df_cn.index
    common_index = list(set(ge_index).intersection(cn_index))
    df_ge = df_ge.loc[common_index]
    df_cn = df_cn.loc[common_index]

    # Order index
    df_ge = df_ge.sort_index()
    df_cn = df_cn.sort_index()
    return df_ge, df_cn

def emulating_original_data():
    """
    Main function to emulate the original data from the curated dataset by ANL.

    This will create new data in the folder ./data/new_data/ with the same format as the original data, 
    saving one dataframe per patient into a csv file with the name of the patient as the name of the file.
    """
    print("Starting Process")
    df_ge = iu.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    df_cn = iu.load_copy_number_data(gene_system_identifier="Gene_Symbol")
    df_ge, df_cn = processing_data(df_ge, df_cn)
    if not os.path.exists('./data/new_data/'):
        os.makedirs('./data/new_data/', exist_ok=True)
    for i, patient in tqdm(enumerate(df_ge.index)):
        gene_exp = df_ge.iloc[i]
        copy_num = df_cn.iloc[i]
        df_patient = pd.concat([gene_exp, copy_num], axis=1)
        df_patient = df_patient.loc[gene_list]
        df_patient.columns = ['gene_expression', 'copy_number']
        df_patient.to_csv(f'./data/new_data/{patient}.csv', sep=',', index=True, header=True)
    print("Finished Process")

def ppi_preprocessing():
    """
    Function to preprocess the PPI network. The only difference is the removal of the genes that are not in the dataset.
    """
    ppi = pd.read_csv('./data/PPI/PPI_network.txt', sep='\t', header=None)
    ppi_new = ppi.copy()
    for gene in not_in_dataset:
        ppi_new = ppi_new[ppi_new[0] != gene]
        ppi_new = ppi_new[ppi_new[1] != gene]
    ppi_new.to_csv('./data/PPI/PPI_network_new.txt', sep='\t', index=False, header=False)

def auc_response():
    """
    Creation of the Pivot table as in the original data, the rows are the chemicals, the columns the cell lines. 
    """
    df_50 = iu.load_single_drug_response_data(source = 'CCLE')
    # From the row improve_chem_id remove the string 'PC_'
    df_50['improve_chem_id'] = df_50['improve_chem_id'].str.replace('PC_', '')
    # Putting in the columns the improve_sample_id and in the rows the improve_chem_id
    # we can do this with a pivot table in pandas.
    df_50_pivot = df_50.pivot(index='improve_chem_id', columns='improve_sample_id', values='auc')
    # Sort the columns by the name of the column
    df_50_pivot = df_50_pivot.sort_index(axis=1)
    df_50_pivot.to_csv('./data/auc_response.csv', sep=',', index=True, header=True)

if __name__ == '__main__':
    # Measure Time
    start = t()
    emulating_original_data()
    ppi_preprocessing()
    auc_response()
    end = t()
    print(f"Time: {end - start} seconds")