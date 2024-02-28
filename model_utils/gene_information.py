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
# from tqdm import tqdm
from time import time as t


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


def emulating_original_data(output_dir, ge, cn):
    """
    Save the omics data. 
    saving one dataframe per patient into a csv file with the name of the patient as the name of the file.
    """
    # Set first column as index
    ge = ge.set_index(ge.columns[0])
    cn = cn.set_index(cn.columns[0])
    if not os.path.exists(output_dir + "_omics_data/"):
        os.makedirs(output_dir + "_omics_data/", exist_ok=True)
        
    # df_ge = iu.load_gene_expression_data(gene_system_identifier="Gene_Symbol", )
    # df_cn = iu.load_copy_number_data(gene_system_identifier="Gene_Symbol")
    # df_ge, df_cn = processing_data(df_ge, df_cn)
    # Note: after this step, the GE and CN have gene symbols as columns and CL names as index. 
    
    # Find intersection genes, and genes that are not in the IMPROVE data.
    gene_list = pd.read_csv('./data/CCLE/gene_list.txt', header=None).values.astype(str).flatten().tolist()
    gene_list = [x for x in gene_list if x in ge.columns]
    genes_not_in_dataset = [x for x in gene_list if x not in ge.columns]
    
    # Write intersection genes into gene_list.txt.
    with open(output_dir + "gene_list.txt", 'w') as file:
        for item in gene_list:
            file.write(str(item) + '\n')

    # Save patient (CL) omics csvs. 
    for i, patient in enumerate(ge.index):
        gene_exp = ge.iloc[i]
        copy_num = cn.iloc[i]
        df_patient = pd.concat([gene_exp, copy_num], axis=1)
        df_patient = df_patient.loc[gene_list]
        df_patient.columns = ['gene_expression', 'copy_number']
        df_patient.to_csv(output_dir + f'_omics_data/{patient}.csv', sep=',', index=True, header=True)

    return genes_not_in_dataset


def ppi_preprocessing(output_dir, genes_not_in_dataset):
    """
    Function to preprocess the PPI network. The only difference is the removal of the genes that are not in the dataset.
    Save the processed PPI to output_dir/PPI/PPI_network_new.txt
    """
    if not os.path.exists(output_dir + "_PPI/"):
        os.makedirs(output_dir + "_PPI/", exist_ok=True)
        
    ppi = pd.read_csv('./data/PPI/PPI_network.txt', sep='\t', header=None)
    ppi_new = ppi.copy()
    for gene in genes_not_in_dataset:
        ppi_new = ppi_new[ppi_new[0] != gene]
        ppi_new = ppi_new[ppi_new[1] != gene]
    ppi_new.to_csv(output_dir + '_PPI/PPI_network_new.txt', sep='\t', index=False, header=False)


def preprocess_omics_data(output_dir, ge, cn):
    """
    Processed data files will be stored in data/IMPROVE_{source}/
    
    """
    print("Starting Process")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    genes_not_in_dataset = emulating_original_data(output_dir, ge, cn)
    ppi_preprocessing(output_dir, genes_not_in_dataset)
    
    print("Finished Process")



if __name__ == '__main__':
    # Measure Time
    start = t()
    preprocess_omics_data("./data_new/IMPROVE_test/", ge, cn)
    end = t()
    print(f"Time: {end - start} seconds")
    