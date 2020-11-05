#put functions in separate file, will do seperate run for HTO data.
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

#insert date at end
date = '82620'
output_text = 'sc_pcqc/Cluster_Analysis_pt5/cluster_evalv' + date + '.txt'
output_plots = 'sc_pcqc/Cluster_Analysis_pt5/cluster_eval_plotsv' + date + '.pdf'
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
preprocess_data = preprocess_data[:-1,:]
ground_truth['count'] = 1
df = pd.DataFrame(preprocess_data)
hto_df.columns = ['barcode_adt','hto_cluster']
hto = pd.merge(new_barcodes, hto_df, on = 'barcode_adt', how = 'outer')
hto['leiden'] = hto['hto_cluster'].map({'Singlet':0,'Doublet':1,'Negative':np.nan})
pd.crosstab(hto['hto_cluster'].fillna('Missing'), ground_truth['leiden'])

#will want to do to two seperate tests: one for detecting doublets and the other for detecting
#cell types
adata = sc.AnnData(X = df.values)
sc.tl.pca(adata,n_comps = 100)

#quick fix for leiden bug
ground_truth['louvain'] = ground_truth['leiden']

columns = ['name', 'trial', 'resolution', 'n_neighbors', 'log_loss','aps','weighted_accuracy']
for value in np.unique(ground_truth['leiden']):
    columns.append('log_loss_cluster_' + str(value))

df_stats = pd.DataFrame(columns = columns)

'''adjust the n_pcs values for evaluating clustering by scree plot/permutation test
'''
for n_pcs in [12,14,17,18]:
    if n_pcs in [12,14]:
        string = 'PC_Scree' + str(n_pcs)
    else:
        string = 'PC_Perm' + str(n_pcs)
    df_stats = cluster_functions.data_sim(adata, string, df_stats, ground_truth, random_state_start = 0, n_pcs = n_pcs)

df_stats.to_csv(‘Cluster_Analysis_pt5/df_stats_pt1.csv’)

for n_pcs in [30,50,70]:
    string = 'PC_Test_' + str(n_pcs)
    df_stats = cluster_functions.data_sim(adata, string, df_stats, ground_truth, random_state_start = 0, n_pcs = n_pcs)
df_stats.groupby('name').median()
df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt1.csv')
#confirm that different pcqc thresholds refer to same PCs.

#will want to adjust parameter values
pca = PC_Cluster(n_pcs = 100)
pca.fit(df.values)
pca.pc_distribution()
pca.pc_stats(thresholds=[0.9,0.95,0.98,0.99,0.995])
#top 14 pcqc pcs - are pcs 0-13
pca.df_pc_stats.sort_values(by = '100.0_Percentile', ascending = False).head(14)
pca.df_pc_stats.sort_values(by = '99.0_Percentile', ascending = False).head(14)
pca.df_pc_stats.sort_values(by = '98.0_Percentile', ascending = False).head(14)
#Both methods select the same PCs 0-4,7-9,11
pca.df_pc_stats.sort_values(by = '95.0_Percentile', ascending = False).head(9)
pca.df_pc_stats.sort_values(by = '90.0_Percentile', ascending = False).head(9)

pca.select_top_pcs(criteria = '100.0_Percentile', n_top_pcs = 14)
reduced_matrix = pca.use_top_pcs()

#good for manually selecting good Kruskal pc's
#may not need this if you use right select_top_pcs criteria
#cols = [0,1,2,3,4,5,6,10]
#reduced_matrix = reduced_matrix[:,cols]
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)
df_stats = cluster_functions.data_sim(adata_pcqc_kruskal, 'PCQC_99.5P_14', df_stats, ground_truth,
                    random_state_start = 0, n_pcs = 0, use_rep = 'X')

df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt1.csv')

pca.select_top_pcs(criteria = '95.0_Percentile', n_top_pcs = 9)
reduced_matrix = pca.use_top_pcs()
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)
df_stats = cluster_functions.data_sim(adata_pcqc_kruskal, 'PCQC_95P_9', df_stats, ground_truth,
                    random_state_start = 0, n_pcs = 0, use_rep = 'X')
df_stats.to_csv('Cluster_Analysis_pt5/df_stats_pt1.csv')

sns.countplot(x = "leiden", data= ground_truth)
plt.title('Ground Truth Cluster Frequencies')
plot_dictionary['ClusterFrequencies'] = plt.gca().get_figure()
plt.close()

sns.boxplot('name', 'log_loss_cluster_11', data=df_stats, boxprops={'facecolor':'None'})
sns.swarmplot('name', 'log_loss_cluster_11', data=df_stats, zorder=.5)
plt.title('Log Loss Cluster 11')
plot_dictionary['Log Loss Cluster 11'] = plt.gca().get_figure()
plt.close()

 #log_loss_cluster_11
#Make some plots, export some text
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



#kruskal Top 25
pca.select_top_pcs(criteria = 'evalue', n_top_pcs = 100)
reduced_matrix = pca.use_top_pcs()
_, kruskal_out = kruskal.compute_best_pcs(reduced_matrix, 15, .8, threshold = 1,
                                    clustering_function = sc.tl.louvain, col_name = 'louvain')
y = np.log10(np.array(kruskal_out)+1e-300)
#depends on answer np.argsort
top_comps = np.argsort(y)[0:25]
reduced_matrix = reduced_matrix[:,top_comps]
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)
df_stats = cluster_functions.data_sim(adata_pcqc_kruskal, 'PCQC_Kruskal_25', df_stats, ground_truth,
                    random_state_start = 0, n_pcs = 0, use_rep = 'X')

df_stats.to_csv('cluster_eval.csv')
#Do One More Test
#good for selecting top pc's by regular method
pca.select_top_pcs(criteria = '100.0_Percentile', n_top_pcs = 13)
reduced_matrix = pca.use_top_pcs()
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)
df_stats = data_sim(adata_pcqc_kruskal, 'PCQC_99.5P_13', df_stats, ground_truth,
                    random_state_start = 0, n_pcs = 0, use_rep = 'X')

sns.boxplot('name', 'log_loss', data=df_stats, boxprops={'facecolor':'None'})
sns.swarmplot('name', 'log_loss', data=df_stats, zorder=.5)
plt.title('Log Loss')
plot_dictionary['Log Loss'] = plt.gca().get_figure()
plt.close()

#plotting other cluster performance if needed, will need to change name of variable
#plt.rcParams['figure.figsize'] = [20, 10]
#sns.boxplot('name', 'log_loss_cluster_Dendritic', data=df_stats, boxprops={'facecolor':'None'})
sns.swarmplot('name', 'log_loss_cluster_Dendritic', data=df_stats, zorder=.5)
plt.title('Log Loss Subcategory')
plot_dictionary['Log Loss Subcategory'] = plt.gca().get_figure()
plt.close()
#save plots to a dictionary?

with PdfPages(output_plots) as pdf:
    for title in plot_dictionary.keys():
        pdf.savefig(plot_dictionary[title])

with open(output_text,'a') as f:
    #automatically closes the file
    for text in text_outputs:
        print(text, file = f)

#ground_truth = pd.read_csv('sample_reweighted_ground_truth.csv')
#ground_truth.columns = ['Unnamed:0','leiden']
#may need to align columns
