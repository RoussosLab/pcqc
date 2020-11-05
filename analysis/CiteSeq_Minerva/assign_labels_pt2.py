import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from numpy.linalg import norm
import scanpy as sc
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import coo_matrix
#need to be able to exclude rows already filtered out.

adt_names = pd.read_csv('features.tsv.gz', header = None)
adt_barcodes = pd.read_csv('barcodes.tsv.gz', header = None)
directory = ''
barcodes_directory = directory + '20190110-cDNA/outs/raw_feature_bc_matrix/barcodes.tsv.gz'
original_barcodes = pd.read_csv(barcodes_directory, header = None)
#'AAACCCAAGAAAGTCT-1'
csv_filter_loc = ''
output_csv_loc = ''
#first find out what rows were filtered
mingene_filter = pd.read_csv(csv_filter_loc,
                            header = 0)

mingene_filter.columns = ['id','keep']
original_barcodes.reset_index(inplace = True)
original_barcodes.columns = ['id','barcode']
mingene_filter = pd.merge(original_barcodes, mingene_filter, on = 'id')
mingene_filter['simple_barcode'] = mingene_filter['barcode'].str[:-2]
adt_barcodes.reset_index(inplace = True)
adt_barcodes.columns = ['adt_index','barcode_adt']

merged_barcodes = pd.merge(adt_barcodes, mingene_filter,
                left_on = 'barcode_adt', right_on = 'simple_barcode', how = 'left')


text_outputs = []
text_outputs.append('1. Merged Barcode Stats')
text_outputs.append('ADT Shape:' + str(adt_barcodes.shape))
text_outputs.append('Initial Gene Filter:' + str(merged_barcodes['keep'].sum()))

final_preprocess_filter = pd.read_csv('sc_pcqc/preprocess_cell_filter.csv',
                            header = 0)
final_preprocess_filter.columns = ['barcode','final_filter']
merged_barcodes_final = pd.merge(merged_barcodes, final_preprocess_filter,
                on = 'barcode', how = 'left')

text_outputs.append('2. Final Merged Barcode Stats')
text_outputs.append('Original QC Shape:' + str(final_preprocess_filter['final_filter'].sum()))
text_outputs.append('New QC Shape with ADT:' + str(merged_barcodes_final['final_filter'].sum()))

merged_barcodes_final.to_csv(output_csv_loc, index = False)
#remaining steps
#1. Read in the ADT Data
#2. Filter out relevant Indices
matrix = pd.read_csv('matrix.mtx.gz', header = 0)
matrix = matrix.iloc[1:]
matrix.columns = ['entry_val']
#reindexed from 1 above
new_vars = ['row_adt','col_bar','value']
for i,var in enumerate(new_vars):
    matrix[var] = matrix['entry_val'].str.split(' ').apply(lambda x: x[i])

valid_bars = merged_barcodes_final.loc[merged_barcodes_final['final_filter'] == True,
                                        'adt_index'].unique()
matrix['col_bar'] = matrix['col_bar'].astype(int)
matrix['row_adt'] = matrix['row_adt'].astype(int)
matrix['value'] = matrix['value'].astype(int)

matrix = matrix.loc[matrix['col_bar'].isin(valid_bars)]
barcode_keys = matrix[['col_bar']].drop_duplicates().reset_index()
barcode_keys['new_col'] = barcode_keys.index
matrix = pd.merge(matrix, barcode_keys[['col_bar','new_col']], on = 'col_bar')

row = matrix['row_adt']
col = matrix['new_col']
value = matrix['value']
sp_matrix = coo_matrix((value, (row, col)), shape=(16, 17555))
#https://github.com/theislab/scanpy/pull/1117
def CLR_transform(df):
    '''
    implements the CLR transform used in CITEseq (need to confirm in Seurat's code)
    https://doi.org/10.1038/nmeth.4380
    '''
    logn1 = np.log(df + 1)
    T_clr = logn1.sub(logn1.mean(axis=1), axis=0) #subtract off column mean from each row
    return T_clr

adt = sc.AnnData(X = sp_matrix.transpose().toarray())
adt.X = CLR_transform(pd.DataFrame(adt.X)).values
#may want to do some preprocessing here
sc.pp.neighbors(adt, n_neighbors = 15)
sc.tl.louvain(adt, random_state = 0, resolution = .4)
#sc.tl.leiden(adt, random_state = 0, resolution = .4)
ground_truth = pd.DataFrame()
#leiden
ground_truth['leiden'] = adt.obs['louvain'].values
ground_truth['leiden'] = adt.obs['leiden'].values
ground_truth['leiden'] = ground_truth['leiden'].astype(int)
ground_truth['count'] = 1
pd.set_option('max_columns', None)
ground_truth.groupby('leiden')['count'].sum()
text_outputs.append('3. Cluster Frequencies')
text_outputs.append(str(ground_truth.groupby('leiden')['count'].sum()))

plot_dictionary = dict()

#stopped here
ground_truth = pd.concat([ground_truth,pd.DataFrame(sp_matrix.transpose().toarray())],axis = 1)
ground_truth.groupby('leiden').median()
text_outputs.append('3. Cluster Stats')
text_outputs.append(str(ground_truth.groupby('leiden').describe()))
sns.heatmap(ground_truth.groupby('leiden').median())
plot_dictionary['Heatmap'] = plt.gca().get_figure()
plt.close()

for i in range(16):
    sns.boxplot(x = 'leiden', y = i, data = ground_truth)
    plt.title('ADT Cluster Distribution ' + str(i))
    plot_dictionary['Boxplot_' + str(i)] = plt.gca().get_figure()
    plt.close()


""" QC to get the right cluster mapping
cluster_mapping = {0:'CD14+ Monocyte', 1: 'CD4 T Cells', 2: 'CD4 T Cells',
                  3: 'NK', 4: 'Mouse/CD34+', 5: 'CD4 T Cells', 6: 'B',
                  7: 'NK', 8: 'CD8', 9: 'CD16', 10: 'Doublets', 11: 'Mouse/CD34+',
                  12: 'Other', 13: 'Other'}
"""

output_plots = ''
output_data = ''
output_text = ''

text_outputs.append('ADT_Names')
text_outputs.append(str(adt_names))
#text_outputs
with PdfPages(output_plots) as pdf:
    for title in plot_dictionary.keys():
        pdf.savefig(plot_dictionary[title])

with open(output_text,'a') as f:
    #automatically closes the file
    for text in text_outputs:
        print(text, file = f)

#ground_truth['truth_names'] = ground_truth['leiden'].replace(cluster_mapping)

ground_truth.to_csv(output_data)

'''
End Here
'''
