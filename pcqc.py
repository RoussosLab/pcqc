import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.linalg import svds
from numpy.linalg import norm
import copy
from matplotlib.backends.backend_pdf import PdfPages

try:
    import scanpy as sc
except:
    pass

class PC_Cluster:
    def __init__(self, n_pcs = 100, solver = 'arpack', v0 = None, observation = 'rows'):
        '''
        Inputs:
        n_pcs: number of principal components to retain
        solver: currently only arpack is supported
        v0: initial starting point for computing eigenvector
        observation: Denote whether observations are rows or columns

        Returns:
        run_PCA object with specified parameters
        '''

        self.n_pcs = n_pcs
        self.solver = solver
        #initial starting point for computing eigenvector
        self.v0 = v0
        self.observation = 'rows'

    def fit(self,matrix):
        '''
        Inputs:
        matrix: NumPY array to perform PCA on
        observation: Denotes whether rows or columns contain observations.

        Output:
        Updated run_PCA object with the following attributes:
        right_PCs: vt, in the SVD decompositon u@s@vt
        left_PCs: u, in the SVD decomposition u@s@vt
        singular_values: the singular values in the SVD decomposition
        total_var: total variance, or square of the Frobenius norm of the matrix
        norm_eigenvalues: the eigenvalues of PCA normalized by total_var.
        var_by_record: vector measuring the total variance for each row.
        '''
        u,s,vt = svds(matrix,k=self.n_pcs,solver=self.solver,v0=self.v0)
        #sort u,s,vt by singular values
        sort_key = np.argsort(-s)
        self.right_PCs = vt[sort_key,:]
        self.left_PCs = u[:,sort_key]
        self.singular_values = s[sort_key]
        self.total_var = norm(matrix, ord = 'fro')**2
        self.norm_eigenvalues = np.power(self.singular_values,2)/self.total_var

        #compute variance by record
        if self.observation == 'rows':
            n_observations = matrix.shape[0]
            var_by_record = np.zeros(n_observations)
            for i in range(n_observations):
                var_by_record[i] = np.dot(matrix[i,:],matrix[i,:])

        else:
            n_observations = matrix.shape[1]
            var_by_record = np.zeros(n_observations)
            for i in range(n_observations):
                var_by_record[i] = np.dot(matrix[:,i],matrix[:,i])

        self.var_by_record = var_by_record


    def pc_distribution(self, n_compute = False):
        '''
        Input:
        n_compute: (Optional).  How many PC's to create a distribution for.  Default is to use all of them.

        Output:
        Creates the attribute df_pca_dist which contains the distribution over all observations
        of the variance explained for a given PC.

        '''
        if not n_compute:
            n_compute = self.n_pcs

        cols = ['PC_Dist_' + str(i) for i in range(self.n_pcs)]
        if self.observation == 'rows':
            #self.left_PCs n_observations x n_pcs
            pc_distribution_data = np.power(self.left_PCs,2)@np.diag(self.norm_eigenvalues)
        else:
            #right_PCs n_pcs x n_observations-so need to transpose it.
            pc_distribution_data = np.power(self.right_PCs.T,2)@np.diag(self.norm_eigenvalues)

        self.df_pca_dist = pd.DataFrame(pc_distribution_data, columns = cols)


    def pc_stats(self, thresholds = [.85,.9,.95]):
        '''
        Inputs:
        matrix: NumPY array to perform PCA on
        thresholds: Identify relevant cutoff thresholds for computing pc_stats

        Output:
        Compute df_pc_stats, adataframe for the threshold percentiles
        for each PC along with the eigenvalue and rank in original ordering.
        '''

        #without transpose, the thresholds will become the indices
        threshold_stats = self.df_pca_dist.apply(lambda x: x.quantile(thresholds)).transpose()
        threshold_stats.columns = [str(100*np.round(t,2)) + '_Percentile' for t in thresholds]
        threshold_stats['evalue'] = self.norm_eigenvalues
        threshold_stats['rank'] = threshold_stats.reset_index().index.astype(float) + 1

        self.df_pc_stats = threshold_stats

    def select_top_pcs(self,criteria, n_top_pcs = 30):
        '''
        Input:
        Criteria (string) column to use from df_pc_stats to select top PCs
        n_top_pcs: Number of top pcs to retain

        Output:
        List of the pc indices to retain.
        '''
        self.best_pcs = self.df_pc_stats.sort_values(by = criteria, ascending = False)['rank'].values[0:n_top_pcs]
        self.best_pcs = self.best_pcs - 1
        self.best_pcs = self.best_pcs.astype(int)

    def use_top_pcs(self, representation = 'left'):
        '''
        Output:
        Returns a matrix that uses only the top pcs
        '''

        if representation == 'full':
            output = self.left_PCs[:,self.best_pcs]@np.diag(self.singular_values[self.best_pcs]) \
                                                    @self.right_PCs[self.best_pcs,:]

        elif representation == 'left':
            output = self.left_PCs[:,self.best_pcs]@np.diag(self.singular_values[self.best_pcs])

        else:
            output = np.diag(self.singular_values[self.best_pcs])@self.right_PCs[self.best_pcs,:]


        return output


    def update_scanpy(self,adata):
        '''
        Input: Annotated data
        Output: Updates scanpy AnnData object with fields for PCA so that you can run clustering as normal.
        '''
        #puts PCA
        adata= create_dictionaries(adata)

        adata.obsm['X_pca'] = self.use_top_pcs()
        if self.observation == 'rows':
            adata.varm['PCs'] =  self.left_PCs[:,self.best_pcs]
        else:
            adata.varm['PCs'] =  self.right_PCs[self.best_pcs,:]

        adata.uns['pca']['variance'] = np.power(self.singular_values[self.best_pcs],2)
        adata.uns['pca']['variance_ratio'] = self.norm_eigenvalues[self.best_pcs]
        return adata

def create_dictionaries(adata):
    #create dictionaries for adata if they do not exist
    try:
        adata.obsm
    except NameError:
        var_exists = False
    else:
        var_exists = True
    if not var_exists:
        adata.obsm = dict()


    try:
        adata.varm
    except NameError:
        var_exists = False
    else:
        var_exists = True
    if not var_exists:
        adata.varm = dict()

    try:
        adata.uns
    except NameError:
        var_exists = False
    else:
        var_exists = True
    if not var_exists:
        adata.uns = dict()


    return adata

def pc_distplot(pc_cluster,cols = np.arange(0,5), upper_limit_factor = 3, upper_limit = False,
               my_function = sns.boxenplot):
    '''
    Inputs:
    my_function: Supports sns.boxplot, sns.boxenplot, sns.violinplot

    '''
    #cols which PCs to select
    df = pc_cluster.df_pca_dist
    df = df.iloc[:,cols]
    melted = pd.melt(df, value_vars=df.columns)
    if not upper_limit:
        upper_limit = upper_limit_factor*melted.groupby('variable')['value'].quantile(.75).max()
    melted.loc[melted['value'] > upper_limit, 'value'] = upper_limit
    my_plot = my_function(data = melted, y = 'variable', x = 'value')
    return my_plot

def pc_distplot_pdf(filename,pc_cluster,vars_per_plot = 10, my_function = sns.boxenplot):
    '''
    Inputs:


    Output:
    PDF with plots for the distribution of variance explained for each of the PC's.
    '''
    start = 0
    factor = np.floor(pc_cluster.df_pca_dist.shape[1]/vars_per_plot)
    end = factor
    counter = 1
    with PdfPages(filename) as pdf:
        while start < pc_cluster.df_pca_dist.shape[1]:
            new_end = np.min([pc_cluster.df_pca_dist.shape[0], end])
            #print(new_end)
            my_plot = pc_distplot(pc_cluster, cols = np.arange(start,new_end),
                                  my_function = my_function)
            plt.title('PC Variance Explained Distribution Page ' + str(counter))
            plt.tight_layout()
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            start = end
            end = end + factor
            counter += 1

def scatter_scree_plot(pca, rank = np.arange(1,51)):
    df = pca.df_pc_stats
    x = df.loc[df['rank'].isin(rank),'rank']
    y = df.loc[df['rank'].isin(rank),'evalue']
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)
    fig.suptitle('Traditional Scree Plot')
    answer = sns.scatterplot(x, y)
    return answer

def dist_var_pc_plot(pca, rank = np.arange(1,51), threshold_subset = False, max_subset_size = 3):
    df = pca.df_pc_stats
    if not threshold_subset:
        threshold_subset = [var for var in df.columns if 'Percentile' in var]
    if len(threshold_subset) > max_subset_size:
        threshold_subset = threshold_subset[0:max_subset_size]
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)
    x = df.loc[df['rank'].isin(rank),'rank']
    for var in threshold_subset:
        y = df.loc[df['rank'].isin(rank),var]
        ax.plot(x, y, label = var, linewidth=3)
    ax.tick_params(axis='both', which='both', length=0, labelsize=8)
    ax.legend(prop={'size': 10})
    plt.xlabel('Rank', fontsize=10)
    plt.ylabel('Variance Explained', fontsize=10)
    fig.suptitle('Distribution of Variance Explained by PC')
    return fig

def sorted_dist_var_pc_plot(pca, rank = np.arange(1,51), threshold_subset = False, max_subset_size = 3,
                           sort_var = 'first'):

    if sort_var == 'first':
        sort_var = pca.df_pc_stats.columns[0]
    new_pca = copy.deepcopy(pca)
    new_pca.df_pc_stats = new_pca.df_pc_stats.sort_values(by = sort_var,
                                                          ascending = False).reset_index()
    new_pca.df_pc_stats['rank'] = new_pca.df_pc_stats.index + 1

    fig = dist_var_pc_plot(new_pca, rank, threshold_subset = threshold_subset,
                           max_subset_size = max_subset_size)
    return fig
