import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from numpy.linalg import norm
import scanpy as sc
from matplotlib.backends.backend_pdf import PdfPages

output_text = 'preprocess.txt'
output_plots = 'preprocess_plots.pdf'
output_data = 'pcqc'


plot_dictionary = dict()
data_file = 'raw_feature_bc_matrix.h5'
adata = sc.read_10x_h5(data_file)
adata.var_names_make_unique()
text_outputs = []

text_outputs.append('1. Import Data')
text_outputs.append(str(adata))


cell_filter1, _ = sc.pp.filter_cells(adata, min_genes=200, inplace = False)
#can save cell_filter1 if desired
initial_filter = pd.DataFrame(cell_filter1)
initial_filter.to_csv('mingene_preprocess_filter.csv')
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
text_outputs.append('2. Filter Cells/Genes')
text_outputs.append(str(adata))

adata.var['mt'] = adata.var_names.str.startswith('MT-')
# annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, inplace=True)
ax = sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
             #plt.gca().get_figure()
plot_dictionary['Data_QC_Violin'] = plt.gca().get_figure()

ax = sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show = False)
plot_dictionary['Data_QC_Violin_2'] = plt.gca().get_figure()
ax = sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show = False)
plot_dictionary['Data_QC_Violin_3'] = plt.gca().get_figure()

cell_filter2 = adata.obs.n_genes_by_counts < 3500
cell_filter3 = adata.obs.pct_counts_mt < 15
final_cell_filter = pd.DataFrame(cell_filter2*cell_filter3)
final_cell_filter.to_csv('preprocess_filter.csv', index = True)
adata = adata[adata.obs.n_genes_by_counts < 3500, :]
adata = adata[adata.obs.pct_counts_mt < 15, :]
text_outputs.append('3. Filter Cells/Genes by Mitochondrial RNA')
text_outputs.append(str(adata))

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
ax = sc.pl.highly_variable_genes(adata, show = False)
plot_dictionary['High_Var_Genes'] = plt.gca().get_figure()

adata.raw = adata
adata = adata[:, adata.var.highly_variable]
text_outputs.append('4. Filter for Highly Variable Genes')
text_outputs.append(str(adata))

sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
#saved it in sc_pcqc
adata.write('preprocess.h5ad')

with PdfPages(output_plots) as pdf:
    for title in plot_dictionary.keys():
        pdf.savefig(plot_dictionary[title])

with open(output_text,'a') as f:
    #automatically closes the file
    for text in text_outputs:
        print(text, file = f)
