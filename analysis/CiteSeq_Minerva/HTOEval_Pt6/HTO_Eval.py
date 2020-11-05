import sys
sys.path.append('./Dim_Reduction_pt3')
sys.path.append('./Cluster_Analysis_pt5')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pcqc import *
import scanpy as sc
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.metrics import average_precision_score, recall_score
from matplotlib.backends.backend_pdf import PdfPages
import cluster_functions
import kruskal_best_pcs as kruskal

date = '82620'
output_text = 'hto_evalv' + date + '.txt'
output_plots = 'hto_eval_plotsv' + date + '.pdf'
text_outputs = []
plot_dictionary = dict()

processed_single_cell_file = 'sc_pcqc/preprocess.h5ad'
#will later want to include HTO tags
ground_truth_file = 'sc_pcqc/ground_truth.csv'
hto_file = 'sc_pcqc/hto.csv'
adata = sc.read(filename = processed_single_cell_file)
ground_truth = pd.read_csv(ground_truth_file)
hto_df = pd.read_csv(hto_file)

#need to get map to see which rows are included
final_preprocess_filter = pd.read_csv('sc_pcqc/preprocess_cell_filter.csv')
barcodes = pd.read_csv('sc_pcqc/adt_cdna_barcode_match.csv')
adata_index = pd.DataFrame(adata.obs.index, columns = ['barcode'])
adata_index = adata_index.reset_index()
#first filter out rows that aren't in data
new_barcodes = pd.merge(adata_index, barcodes, on = 'barcode', how = 'left')
new_barcodes = new_barcodes.loc[new_barcodes['final_filter'] == True]
preprocess_data = adata.X[new_barcodes['index'],:]
ground_truth = ground_truth.rename(columns = {'Unnamed: 0': 'GroundTruthId'})
#filter out cluster ids that arent in data
new_barcodes = new_barcodes.reset_index(drop = True)
preprocess_data = preprocess_data[np.argsort(new_barcodes['adt_index']),:]
#address index but from assign_labels.py
ground_truth['count'] = 1
hto_df.columns = ['barcode_adt','hto_cluster']
hto = pd.merge(new_barcodes, hto_df, on = 'barcode_adt', how = 'outer')
hto['leiden'] = hto['hto_cluster'].map({'Singlet':0,'Doublet':1,'Negative':np.nan})
preprocess_data = preprocess_data[hto['leiden'].notnull(),:]
#preprocess_data = preprocess_data[:-1,:]
df = pd.DataFrame(preprocess_data)
hto_truth = hto.loc[hto['leiden'].notnull()].reset_index(drop = True)

adata = sc.AnnData(X = df.values)
sc.tl.pca(adata,n_comps = 100)

#quick fix for leiden bug
hto_truth['louvain'] = hto_truth['leiden']
hto_truth['count'] = 1

columns = ['name', 'trial', 'resolution', 'n_neighbors', 'log_loss','aps','weighted_accuracy']
for value in np.unique(hto_truth['leiden']):
    columns.append('log_loss_cluster_' + str(value))

df_stats = pd.DataFrame(columns = columns)

'''adjust the n_pcs values for evaluating clustering by scree plot/permutation test
'''

for n_pcs in [12,14,17,18]:
    if n_pcs in [12,14]:
        string = 'PC_Scree' + str(n_pcs)
    else:
        string = 'PC_Perm' + str(n_pcs)
    df_stats = cluster_functions.data_sim(adata, string, df_stats, hto_truth, random_state_start = 0, n_pcs = n_pcs)

df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt2HTO.csv')

for n_pcs in [30,50,70]:
    string = 'PC_Test_' + str(n_pcs)
    df_stats = cluster_functions.data_sim(adata, string, df_stats, hto_truth, random_state_start = 0, n_pcs = n_pcs)
df_stats.groupby('name').median()
df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt2HTO.csv')

pca = PC_Cluster(n_pcs = 100)
pca.fit(df.values)
pca.pc_distribution()
pca.pc_stats(thresholds=[0.9,0.95,0.98,0.99,0.995])

#100.0 Criteria is the same as PC_Scree_14

pca.select_top_pcs(criteria = '95.0_Percentile', n_top_pcs = 9)
reduced_matrix = pca.use_top_pcs()
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)
df_stats = cluster_functions.data_sim(adata_pcqc_kruskal, 'PCQC_95P_9', df_stats, hto_truth,
                    random_state_start = 0, n_pcs = 0, use_rep = 'X')
df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt2HTO.csv')

sns.countplot(x = "leiden", data= hto_truth)
plt.title('Ground Truth Cluster Frequencies')
plot_dictionary['ClusterFrequencies'] = plt.gca().get_figure()
plt.close()

sns.boxplot('name', 'log_loss', data=df_stats, boxprops={'facecolor':'None'})
sns.swarmplot('name', 'log_loss', data=df_stats, zorder=.5)
plt.title('Log Loss')
plot_dictionary['Log Loss'] = plt.gca().get_figure()
plt.close()

sns.boxplot('name', 'weighted_accuracy', data=df_stats, boxprops={'facecolor':'None'})
sns.swarmplot('name', 'weighted_accuracy', data=df_stats, zorder=.5)
plt.title('Weighted Accuracy')
plot_dictionary['Weighted_Accuracy'] = plt.gca().get_figure()
plt.close()

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
text_outputs.append(str(df_stats.groupby(['name']).median()))
text_outputs.append(str(df_stats.groupby(['name']).mean()))

with PdfPages(output_plots) as pdf:
    for title in plot_dictionary.keys():
        pdf.savefig(plot_dictionary[title])

with open(output_text,'a') as f:
    #automatically closes the file
    for text in text_outputs:
        print(text, file = f)
