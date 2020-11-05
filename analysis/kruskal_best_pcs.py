import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.linalg import svds

def extract_group_samples(df, col, group_id = 'Clusters'):
    group_list = []
    for name, group in df.groupby(group_id):
        group_list.append(group[col].values)
    return group_list

def pc_kruskal_test(df,col_names = np.arange(30),group_id = 'Clusters', threshold = 1e-3):
    #assumes df has clusters column
    #and columns for PC's are labeled from 0 up
    kruskal_test = []
    for col in col_names:
        sample_list = extract_group_samples(df,col,group_id)
        _, pvalue = stats.kruskal(*sample_list)
        kruskal_test.append(pvalue)
    best_pcs = [i for i,val in enumerate(kruskal_test) if val < threshold]
    return kruskal_test,best_pcs

def choose_best_pcs(df, cols):
    tiny_matrix = df[cols].values
    return tiny_matrix

def permutation_test(matrix,trials = 100, n_pcs = 100):
    n_cols = matrix.shape[1]
    df = pd.DataFrame(columns = np.arange(n_pcs))
    current_shape = 0
    for i in range(trials):
        print(i)
        for j in range(n_cols):
            matrix[:,j] = np.random.permutation(matrix[:,j])

        #compute PCA
        _,s,_ = svds(matrix,k=100)
        #evalues = np.power(s,2)
        df.loc[current_shape] = s
        current_shape += 1
    return df




#kruskal_test, best_pcs = pc_kruskal_test(df_pca)
