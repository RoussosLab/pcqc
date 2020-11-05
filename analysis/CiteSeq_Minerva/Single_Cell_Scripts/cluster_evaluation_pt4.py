#put functions in separate file
import sys
sys.path.append('../pcqc')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pcqc import *
import scanpy as sc
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.metrics import average_precision_score, recall_score
import cluster_functions

output_text = 'cluster_eval.txt'
output_plots = 'cluster_eval_plots.pdf'
text_outputs = []
plot_dictionary = dict()

processed_single_cell_file = ''
ground_truth_file = ''
adata = sc.read(filename = processed_single_cell_file)
ground_truth = pd.read_csv(ground_truth_file)
ground_truth['count'] = 1
df = pd.DataFrame(adata.X)

adata = sc.AnnData(X = df.values)
sc.tl.pca(adata,n_comps = 100)

columns = ['name', 'trial', 'resolution', 'n_neighbors', 'log_loss','aps','weighted_accuracy']
for value in np.unique(ground_truth['leiden']):
    columns.append('log_loss_cluster_' + str(value))

df_stats = pd.DataFrame(columns = columns)

'''adjust the n_pcs values for evaluating clustering by scree plot/permutation test
'''
for n_pcs in [5,6,7,12,19]:
    if n_pcs in [5,6,7,12]:
        string = 'PC_Scree_Perm' + str(n_pcs)
    else:
        string = 'PC_Perm_Inclusive' + str(n_pcs)
    df_stats = data_sim(adata, string, df_stats, ground_truth, random_state_start = 0, n_pcs = n_pcs)

df_stats.groupby('name').median()

#will want to adjust parameter values
pca = pcqc.PC_Cluster(n_pcs = 100)
pca.fit(df.values)
pca.pc_distribution()
pca.pc_stats(thresholds=[0.9,0.95,0.98,0.99,0.995])
pca.select_top_pcs(criteria = 'evalue', n_top_pcs = 20)
reduced_matrix = pca.use_top_pcs()

#good for manually selecting good Kruskal pc's
#may not need this if you use right select_top_pcs criteria
cols = [0,1,2,3,4,5,6,10]
reduced_matrix = reduced_matrix[:,cols]
adata_pcqc_kruskal = sc.AnnData(X = reduced_matrix)

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
plt.rcParams['figure.figsize'] = [20, 10]
sns.boxplot('name', 'log_loss_cluster_Dendritic', data=df_stats, boxprops={'facecolor':'None'})
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
