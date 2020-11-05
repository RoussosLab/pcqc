#need the pcqc and kruskal_best_pcs files
#Results:
#Top 14 Scree, Permutation, 100.0 0_Percentile
#Top 9 PCQC different thresholds 95.0
#Kruskal Top 25
import sys
package_location = "./Dim_Reduction_pt3"
sys.path.append(package_location)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from numpy.linalg import norm
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages
import pcqc
import kruskal_best_pcs as kruskal

output_text = 'sc_pcqc/dim_reduce.txt'
output_plots = 'sc_pcqc/dim_reduce_plots.pdf'
text_outputs = []
plot_dictionary = dict()
adata = sc.read(filename = 'sc_pcqc/preprocess.h5ad')
data = adata.X


pca = pcqc.PC_Cluster(n_pcs = 100)
pca.fit(data)
pca.norm_eigenvalues[0:20]
pca.singular_values[0:40]

text_outputs.append('1. Top PCs')
text_outputs.append('Top Eigenvalues: ' + str(pca.norm_eigenvalues[0:20]))
text_outputs.append('Top Singular Values:  ' + str(pca.singular_values[0:40]))

pca.pc_distribution()
pca.df_pca_dist.head()

ax = pcqc.pc_distplot(pca, cols = np.arange(0,10))
plot_dictionary['PCQC Distplot'] = plt.gca().get_figure()
plt.close()

#2. Perform Scree Plots
pca.pc_stats(thresholds=[0.9,0.95,0.98,0.99,0.995])
#top 14 according to scree plot
gap_stats = kruskal.compute_gap(pca.df_pc_stats,'evalue')
gap_stats[0:10]
text_outputs.append('2. Scree Plot Knee')
text_outputs.append(str(gap_stats[0:20]))
ax = sns.boxplot(gap_stats['gaps'])
plt.title('Gap Stats, Scree Plot')
plot_dictionary['Gap Stats, Scree Plot'] = plt.gca().get_figure()
plt.close()
pcqc.scatter_scree_plot(pca)
plot_dictionary['Scree Plot'] = plt.gca().get_figure()
plt.close()
#might want to adjust rank
pcqc.scatter_scree_plot(pca, rank = np.arange(5,51))
plot_dictionary['Scree Plot_2'] = plt.gca().get_figure()
plt.close()

#3. PCQC Plots
#do some exploration for plotting the pcqc scree plots
1e-6*pca.df_pc_stats.sort_values(by = '100.0_Percentile',
                                 ascending = False).head(20)

1e-6*pca.df_pc_stats.sort_values(by = '95.0_Percentile',
                                    ascending = False).head(20)

1e-6*pca.df_pc_stats.sort_values(by = '99.0_Percentile',
                                    ascending = False).head(20)

fig = pcqc.dist_var_pc_plot(pca, rank = np.arange(5,35), threshold_subset = ['98.0_Percentile',
                                                                        '99.0_Percentile',
                                                                        '100.0_Percentile'])
plot_dictionary['PCQC Plot'] = plt.gca().get_figure()
plt.close()

fig = pcqc.sorted_dist_var_pc_plot(pca, rank = np.arange(10,50), sort_var = '100.0_Percentile',
                             threshold_subset =                       ['98.0_Percentile',
                                                                        '99.0_Percentile',
                                                                        '100.0_Percentile'])

plot_dictionary['Sorted PCQC Plot'] = plt.gca().get_figure()
plt.close()

#will want to change threshold
#much more pronounceed at Top 14, Maybe Top 15
gap_stats = kruskal.compute_gap(pca.df_pc_stats,'100.0_Percentile')
gap_stats[0:12]

text_outputs.append('3. Scree Plot PCQC Knee 99.5')
text_outputs.append(str(gap_stats[0:10]))

gap_stats = kruskal.compute_gap(pca.df_pc_stats,'99.0_Percentile')
gap_stats[0:12]

text_outputs.append('3a. Scree Plot PCQC Knee 99')
text_outputs.append(str(gap_stats[0:10]))

gap_stats = kruskal.compute_gap(pca.df_pc_stats,'98.0_Percentile')
gap_stats[0:12]
text_outputs.append('3b. Scree Plot PCQC Knee 98')
text_outputs.append(str(gap_stats[0:12]))

#Top 9
gap_stats = kruskal.compute_gap(pca.df_pc_stats,'95.0_Percentile')
gap_stats[0:12]
text_outputs.append('3c. Scree Plot PCQC Knee 95')
text_outputs.append(str(gap_stats[0:10]))

#Top 9
gap_stats = kruskal.compute_gap(pca.df_pc_stats,'90.0_Percentile')
gap_stats[0:12]
text_outputs.append('3d. Scree Plot PCQC Knee 90')
text_outputs.append(str(gap_stats[0:10]))

#4. Kruskal Selection PC's.  Use Top 25-31 depending on threshold
pca.select_top_pcs(criteria = 'evalue', n_top_pcs = 100)
reduced_matrix = pca.use_top_pcs()
_, kruskal_out = kruskal.compute_best_pcs(reduced_matrix, 15, .8, threshold = 1,
                                    clustering_function = sc.tl.louvain, col_name = 'louvain')
y = np.log10(np.array(kruskal_out)+1e-300)
ax = plt.scatter(x = np.arange(len(y)), y = np.sort(y))
plt.title('Kruskal Test')
plot_dictionary['Kruskal Test'] = plt.gca().get_figure()
plt.close()
gap_stats = kruskal.compute_gap(pd.DataFrame(-1*np.sort(y),columns = ['kruskal']),'kruskal')
gap_stats[0:10]
text_outputs.append('4. Gap Stats Kruskal')
text_outputs.append(str(gap_stats[0:10]))

#5. Kruskal Stratified, Top 27 in Kruskal Ranking
pca.select_top_pcs(criteria = 'evalue', n_top_pcs = 100)
reduced_matrix = pca.use_top_pcs()
_, kruskal_out = kruskal.compute_best_pcs_strat(reduced_matrix, 15, .8, threshold = 1,
                                            cluster_function = sc.tl.louvain, col_name = 'louvain')
y = np.log10(np.array(kruskal_out)+1e-300)
ax = plt.scatter(x = np.arange(len(y)), y = np.sort(y))
plt.title('Kruskal Stratified Test')
plot_dictionary['Kruskal Stratified Test'] = plt.gca().get_figure()
plt.close()
gap_stats = kruskal.compute_gap(pd.DataFrame(-1*np.sort(y),columns = ['kruskal']),'kruskal')
gap_stats[0:10]
text_outputs.append('5. Gap Stats Kruskal Stratified')
text_outputs.append(str(gap_stats[0:10]))

#6. Kruskal Stratified with PCQC, Top 27
pca.select_top_pcs(criteria = '100.0_Percentile', n_top_pcs = 15)
reduced_matrix = pca.use_top_pcs()
_, kruskal_out = kruskal.compute_best_pcs_strat(reduced_matrix, 15, .8, threshold = 1,
                                            cluster_function = sc.tl.louvain, col_name = 'louvain')
y = np.log10(np.array(kruskal_out)+1e-300)
ax = plt.scatter(x = np.arange(len(y)), y = np.sort(y))
plt.title('Kruskal PCQC Stratified Test')
plot_dictionary['Kruskal PCQC Stratified Test'] = plt.gca().get_figure()
plt.close()
gap_stats = kruskal.compute_gap(pd.DataFrame(-1*np.sort(y),columns = ['kruskal']),'kruskal')
gap_stats[0:10]
text_outputs.append('6. Gap Stats PCQC Kruskal Stratified')
text_outputs.append(str(gap_stats[0:10]))
#downside is ratios become unstable

#7. Permutation Test, puts it at 17.
perm_test = kruskal.permutation_test(data, trials = 20)
sorted_medians = -1*np.sort(-1*perm_test.median())
e_ratio = kruskal.compute_e_ratio(pca.singular_values,sorted_medians)
e_ratio[0:10]
text_outputs.append('7. Permutation Test')
text_outputs.append(str(e_ratio[0:10]))
sorted_svalues = -1*np.sort(-1*perm_test.max())
np.sum(pca.singular_values > sorted_svalues)
ax = plt.scatter(x = np.arange(5,100), y = pca.singular_values[5:] - sorted_svalues[5:])
plt.title('Permutation Test')
plot_dictionary['Permutation Test'] = plt.gca().get_figure()
plt.close()

#print output to files
with PdfPages(output_plots) as pdf:
    for title in plot_dictionary.keys():
        pdf.savefig(plot_dictionary[title])

with open(output_text,'a') as f:
    #automatically closes the file
    for text in text_outputs:
        print(text, file = f)
