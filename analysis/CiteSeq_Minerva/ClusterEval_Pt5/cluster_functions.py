import sys
sys.path.append("sc_pcqc")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pcqc import *
import scanpy as sc
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.metrics import average_precision_score, recall_score


def data_sim(adata, sim_name, df_stats, ground_truth, res_params = np.arange(.4,2,.4),
             n_neighbor_params = np.arange(10,30,5), trials = 20, random_state_start = 0, save = False,
            n_pcs = None, use_rep = None, save_directory = '', cluster_function = sc.tl.louvain,
            col_name = 'louvain'):
    random_state = random_state_start
    index = df_stats.shape[0]
    y = ground_truth[col_name]
    unique_values = np.unique(y)
    weights = ground_truth.groupby(col_name)['count'].transform(lambda x: 1/x.count()).values
    #create new directory to save results
    directory = save_directory + '/' + sim_name
    if save:
        os.mkdir(directory)

    for n_neighbors in n_neighbor_params:
        sc.pp.neighbors(adata,n_neighbors = n_neighbors, use_rep = use_rep, n_pcs = n_pcs)
        for res_param in res_params:
            for trial in range(trials):
                print('Trial: ' + str(np.round(trial,2)) + ' Res: ' + str(np.round(res_param,2)) + ' Nbrs: ' + str(np.round(n_neighbors,2)) + '   ',
                      end = '\r')
                #compute simulation with metrics
                cluster_function(adata, resolution = res_param, random_state = random_state)
                X = pd.get_dummies(adata.obs[col_name])
                predictions, metrics = compute_model_loss(X,y,weights)
                submetrics = compute_log_loss_by_class(predictions,y)
                new_row = [sim_name, trial, res_param, n_neighbors,
                                       metrics['log_loss'], metrics['average_precision_score'],
                                      metrics['weighted_accuracy']] + submetrics

                df_stats.loc[index] = new_row

                #if we want to save the predictions
                if save:
                    file_location = directory + '/' + 'res_' + str(res_param) + 'n_nbrs_' + str(n_neighbors) \
                                + 'trial_' + str(trial) + '.npy'
                    np.save(file_location,predictions)

                index += 1
                random_state += 1
    return df_stats

def compute_log_loss_by_class(predictions,y):
    output = []
    unique_values = np.unique(y)
    for response in unique_values:
        filtered_rows = np.where(y == response)[0]
        subset_log_loss = log_loss(y[filtered_rows], predictions[filtered_rows,:], labels = unique_values)
        output.append(subset_log_loss)
    return output

def compute_model_loss(X,y,weights):
    dummies_y = pd.get_dummies(y).values
    Bayes = MultinomialNB(fit_prior = False)
    Bayes.fit(X,y)
    predictions = Bayes.predict_proba(X)
    loglo = log_loss(y, predictions, sample_weight = weights)
    aps = average_precision_score(dummies_y, predictions, average = 'macro')
    weighted_accuracy = Bayes.score(X,y,sample_weight = weights)
    metrics = {'log_loss':loglo,'average_precision_score': aps,
               'weighted_accuracy':weighted_accuracy}
    return predictions, metrics

def prob_confusion_matrix(probs,ground_truth):
    #probabilistic confusion matrix that enforces balanced class sizes
    df = pd.DataFrame(probs)
    df['clusters'] = ground_truth
    confusion_matrix = df.groupby('clusters').mean()
    return confusion_matrix
