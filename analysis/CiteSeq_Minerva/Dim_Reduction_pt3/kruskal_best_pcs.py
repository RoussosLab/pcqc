import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.linalg import svds
import scanpy as sc


def strat_pc_kruskal_test(df, col_names = np.arange(30), group_id = 'Clusters',
                        threshold = 1e-3, sample_n = 100):

    #sample bootstrap 100 records from each group,
    new_df_list = []
    for group in df[group_id].unique():
        sub_df = df.loc[df[group_id] == group].reset_index(drop = True)
        new_sample = sub_df.sample(n = sample_n, replace = True)
        new_df_list.append(new_sample)
    new_df = pd.concat(new_df_list).reset_index(drop = True)
    kruskal_test, best_pcs = pc_kruskal_test(new_df, col_names = col_names,
                            group_id = group_id, threshold = threshold)
    return kruskal_test, best_pcs

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

def compute_gap(df, var_name):
    nrows = df.shape[0]
    df = df.sort_values(by = var_name,ascending = False).reset_index()
    ratios = np.divide(df.loc[0:nrows-2,var_name].values,df.loc[1:nrows-1,var_name].values)
    top_args = np.argsort(-1*ratios)
    top_gaps = -1*np.sort(-1*ratios)
    gap_stats = pd.DataFrame(top_args, columns = ['args'])
    gap_stats['gaps'] = top_gaps
    return gap_stats

def compute_best_pcs(reduced_matrix, n_neighbors, resolution, threshold = 1e-6, max_discard = 20,
                    clustering_function = sc.tl.leiden,col_name = 'leiden'):
    #max discard, max number of pcs that are discarded in a single round
    df = pd.DataFrame(reduced_matrix)
    finished = False
    index = 0
    while not finished:
        adata = sc.AnnData(X = reduced_matrix)
        sc.pp.neighbors(adata, n_neighbors = n_neighbors, use_rep = 'X')
        clustering_function(adata, resolution = resolution, random_state = np.random.randint(100))
        df['Clusters'] = adata.obs[col_name].values
        valid_cols = [col for col in df.columns if col != 'Clusters']
        kruskal_test, best_pcs = pc_kruskal_test(df,valid_cols, threshold = threshold)
        candidates_to_elim = np.where(np.array(kruskal_test) > threshold)
        #get max discard lowest values
        #print(kruskal_test)
        #print(candidates_to_elim)

        lowest_values = np.argsort(-1*np.array(kruskal_test))[0:max_discard]
        #print(lowest_values)
        to_discard = []
        for value in lowest_values:
            if kruskal_test[value] > threshold:
                to_discard.append(value)
        #print(to_discard)
        to_keep = [col for i,col in enumerate(df.columns) if i not in to_discard]
        if len(to_discard) == 0:
            finished = True
        else:
            df = df[to_keep]
            reduced_matrix = df.values

    return df,kruskal_test

def compute_e_ratio(svalues,perm):
    answer = np.divide(svalues,perm)
    df = pd.DataFrame(answer,columns = ['E_Ratio'])
    return df

def compute_best_pcs_strat(reduced_matrix, n_neighbors, resolution, threshold = 1e-6, max_discard = 20,
                        cluster_function = sc.tl.leiden, col_name = 'leiden'):
    #max discard, max number of pcs that are discarded in a single round

    df = pd.DataFrame(reduced_matrix)
    finished = False
    index = 0
    while not finished:
        adata = sc.AnnData(X = reduced_matrix)
        sc.pp.neighbors(adata, n_neighbors = n_neighbors, use_rep = 'X')
        cluster_function(adata, resolution = resolution, random_state = np.random.randint(100))
        df['Clusters'] = adata.obs[col_name].values
        valid_cols = [col for col in df.columns if col != 'Clusters']
        kruskal_test, best_pcs = strat_pc_kruskal_test(df,col_names = valid_cols, threshold = threshold)
        candidates_to_elim = np.where(np.array(kruskal_test) > threshold)
        #get max discard lowest values
        #print(kruskal_test)
        #print(candidates_to_elim)

        lowest_values = np.argsort(-1*np.array(kruskal_test))[0:max_discard]
        #print(lowest_values)
        to_discard = []
        for value in lowest_values:
            if kruskal_test[value] > threshold:
                to_discard.append(value)
        #print(to_discard)
        to_keep = [col for i,col in enumerate(df.columns) if i not in to_discard]
        if len(to_discard) == 0:
            finished = True
        else:
            df = df[to_keep]
            reduced_matrix = df.values

    return df,kruskal_test



#kruskal_test, best_pcs = pc_kruskal_test(df_pca)
