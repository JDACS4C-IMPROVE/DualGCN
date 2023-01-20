"""
To validate results in paper. 

Figure:
https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04664-4/figures/2

"""

import pandas as pd
from scipy.stats import pearsonr

results = pd.read_csv("../log/result_DualGCNmodel.csv")
drug_list = results.drug_id.unique()
print(drug_list)
cancer_list = results.cancer_type.unique()
print(cancer_list)

for each in cancer_list:
    selected_results = results.loc[results.cancer_type == each]
    pcorr = pearsonr(selected_results.IC50, selected_results.IC50_predicted)
    print(each, pcorr[0])

score_list = []
for each in drug_list:
    selected_results = results.loc[results.drug_id == each]
    pcorr = pearsonr(selected_results.IC50, selected_results.IC50_predicted)
    score_list.append([each, pcorr[0]])

df = pd.DataFrame(score_list)
print(df.sort_values(1))
