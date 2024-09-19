import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Data:
data_val = './out_model/gCSI/split_0/val_y_data_predicted.csv'
history = './out_model/gCSI/split_0history.txt'

# Read data:
df_val = pd.read_csv(data_val)
df_val = df_val[['improve_sample_id','auc_true', 'auc_pred']]

# Regression plot:
plt.figure(figsize=(8, 8))
plt.scatter(df_val['auc_true'], df_val['auc_pred'], alpha=0.5)
# Put a line on the plot:
plt.plot([0.4, 1], [0.4, 1], color='red', lw=2)
plt.title('Regression plot')
plt.xlabel('True AUC')
plt.ylabel('Predicted AUC')
plt.savefig('regression_plot.png')
