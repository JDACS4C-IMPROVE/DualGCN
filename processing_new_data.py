import os
import improve_utils as iu
import pandas as pd
from tqdm import tqdm

gene_list = pd.read_csv('./data/CCLE/gene_list.txt', header= None).values.astype(str).flatten().tolist()
not_in_dataset = ['SEPT6', 'SEPT5', 'FGFR1OP', 'H3F3A', 'C15orf65', 'SEPT9', 'H3F3B', 'CARS']

for gene in not_in_dataset:
    gene_list.remove(gene)


df_ge = iu.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
df_cn = iu.load_copy_number_data(gene_system_identifier="Gene_Symbol")

def processing_data(df_ge, df_cn):
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

def emulating_original_data(df_ge, df_cn):
    print("Starting Process")
    df_ge, df_cn = processing_data(df_ge, df_cn)
    if not os.path.exists('./data/new_data/'):
        os.makedirs('./data/new_data/', exist_ok=True)
    for i, patient in tqdm(enumerate(df_ge.index)):
        gene_exp = df_ge.iloc[i]
        copy_num = df_cn.iloc[i]
        df_patient = pd.concat([gene_exp, copy_num], axis=1)
        df_patient = df_patient.loc[gene_list]
        df_patient.columns = ['gene_expression', 'copy_number']
        df_patient.to_csv(f'./data/new_data/{patient}.txt', sep='\t', index=True, header=True)
    print("Finished Process")

def ppi_preprocessing():
    ppi = pd.read_csv('./data/PPI/PPI_network.txt', sep='\t', header=None)

    ppi_new = ppi.copy()
    for gene in not_in_dataset:
        ppi_new = ppi_new[ppi_new[0] != gene]
        ppi_new = ppi_new[ppi_new[1] != gene]
    ppi_new.to_csv('./data/PPI/PPI_network_new.txt', sep='\t', index=False, header=False)

if __name__ == '__main__':
    emulating_original_data(df_ge, df_cn)
    ppi_preprocessing()