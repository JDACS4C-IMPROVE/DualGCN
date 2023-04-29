
import os
 
def path_function(data_dir):
    """
    Document this (Save for later)
    """
    Drug_info_file = os.path.join(data_dir, 'drug/1.Drug_listMon Jun 24 09_00_55 2019.csv')
    Cell_line_info_file = os.path.join(data_dir, 'CCLE/Cell_lines_annotations_20181226.txt')
    Drug_feature_file = os.path.join(data_dir, 'drug/drug_graph_feat')
    Cancer_response_exp_file = os.path.join(data_dir, 'CCLE/GDSC_IC50.csv')
    # Hard coded PPI file (later):
    PPI_file = os.path.join(data_dir, 'PPI/PPI_network.txt')
    selected_info_common_cell_lines = os.path.join(data_dir, 'CCLE/cellline_list.txt')
    selected_info_common_genes = os.path.join(data_dir, 'CCLE/gene_list.txt')
    ##
    celline_feature_folder = os.path.join(data_dir, 'CCLE/omics_data')
    TCGA_label_set = ["ALL","BLCA","BRCA","DLBC","LIHC","LUAD",
                      "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                      "LUSC","MM","NB","OV","PAAD","SCLC","SKCM",
                      "STAD","THCA",'COAD/READ','SARC','UCEC','MESO', 'PRAD']
    # Return everything in a dictionary
    return {'Drug_info_file': Drug_info_file,
            'Cell_line_info_file': Cell_line_info_file,
            'Drug_feature_file': Drug_feature_file,
            'Cancer_response_exp_file': Cancer_response_exp_file,
            'PPI_file': PPI_file,
            'selected_info_common_cell_lines': selected_info_common_cell_lines,
            'selected_info_common_genes': selected_info_common_genes,
            'celline_feature_folder': celline_feature_folder,
            'TCGA_label_set': TCGA_label_set}    
    
def output_paths(log):
    file_path_pcc_log = os.path.join(log, 'pcc_DualGCNmodel_inference.log')
    file_path_spearman_log= os.path.join(log, 'spearman_DualGCNmodel_inference.log')
    file_path_rmse_log = os.path.join(log, 'rmsd_DualGCNmodel_inference.log')
    file_path_csv = os.path.join(log, 'result_DualGCNmodel_inference.csv')
    return {'file_path_pcc_log': file_path_pcc_log,
            'file_path_spearman_log': file_path_spearman_log,
            'file_path_rmse_log': file_path_rmse_log,
            'file_path_csv': file_path_csv}