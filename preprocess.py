"""
5/16/2023

A script to read the omics data and save:
    Omics data -> a csv for each CL (patient)
    Molecular graph -> a hickle for each drug
    Gene names, selected PPI -> txt files. 

Timing recorded with $time on Lambda0
    real    0m31.718s
    user    0m23.027s
    sys     0m2.369s

"""
    
from process_gen_mol_graph import improve_utils_to_hickle
from model_utils.gene_information import preprocess_omics_data
from improve_utils import load_single_drug_response_data, load_single_drug_response_data_v2

source = "CTRPv2"  # Can be turned into CLI args later. 
output_dir = "./data/IMPROVE_{}/".format(source)

responses = load_single_drug_response_data_v2(source = 'CTRPv2', split_file_name='CTRPv2_split_0_train.txt', y_col_name='auc')
improve_utils_to_hickle(responses, output_dir)

preprocess_omics_data(output_dir)
