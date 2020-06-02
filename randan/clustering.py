#TODO: handle binary str variables (as in logistic regression)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from IPython.display import display

from sklearn.cluster import KMeans as KM_Base
from sklearn.metrics import silhouette_samples, silhouette_score

from .comparison_of_central_tendency import ANOVA
from scipy.spatial.distance import pdist, squareform


class KMeans:
    """
    Class to perform k-means cluster analysis.
    
    Parameters
    ----------
    n_clusters : int 
        Number of clusters in a solution

    Attributes
    ----------
    n_iters : int
        Number of iterations the algorithm used
    is_converged : bool
        Whether the algorithm was converged
    cluster_centers : pd.DataFrame
        Centers of obtained clusters
    inertia : float
        Sum of squared distances of observations to their cluster center
    """
    def __init__(
        self,
        n_clusters=None,
    ):
        if n_clusters is None:
            raise ValueError('Please specify a number of clusters.')
        
        self.n_clusters = n_clusters
        
    def fit(
        self,
        data,
        variables=None,
        max_iter=100,
        tolerance=0.0001,
        random_state=1,
        show_results=True,
        n_decimals=3,
        plot_clusters=True,
        plot_silhouette=False
    ):
        """
        Fit a model to the given data.
        
        Parameters
        ----------
        data : pd.DataFrame 
            Data to fit a model
        variables : list 
            Names of variables from data that should be used in a cluster model.
            If not specified, all variables from data are used. Variables should have a numeric dtype.
        max_iter : int 
            Maximum number of iterations after which the algorithm is forced to stop
        tolerance : float 
            Minimum change in clusters' centers after which the algorithm is forced to stop
        random_state : int 
            An initial point of a random numbers generator to assure results' reproducibility
        show_results : bool 
            Whether to show results of analysis
        n_decimals : int 
            Number of digits to round results when showing them 
        plot_clusters : bool 
            Whether to visualize clusters in a dimension of two most influential variables 
        plot_silhouette : bool 
            Whether to visualize silhouette coefficient (looks like in SPSS' Two-Step Clusters)

        Returns
        -------
        self
            The current instance of the KMeans class
        """
        self._data = data.copy()
        
        if variables is None:
            data = data.dropna().copy()
            variables = list(data.columns)
        else:
            if not isinstance(variables, list):
                raise TypeError(f'Variables should be passed as list. {type(variables)} was passed instead.')
            
            data = data[variables].dropna().copy()
        
        self._integer_type_vars = [var for var in variables if (data[var].apply(round)!=data[var]).sum()==0]
            
        self._observation_idx = list(data.index)
        self._variables = list(data.columns)
        self._cluster_idx = [i+1 for i in range(self.n_clusters)]
        
        self._model = KM_Base(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10, 
            max_iter=max_iter,
            tol=tolerance,
            algorithm='full',
            random_state=random_state
        ).fit(data)
        
        self.n_iters = self._model.n_iter_
        self.is_converged = True if self.n_iters < max_iter else False
        self.cluster_centers = self.get_cluster_centers(round_discrete=False)
        self.inertia = self._model.inertia_
        self._predictions = self.transform(self._data, add_to_data=True)        
        
        if show_results:
            self.show_results(n_decimals)
        
        if plot_silhouette:
            print('------------------\n')
            print('Clusters quality')
            self.get_silhouette_plot()
            print("""This plot is based on the average silhouette score, 
which can be obtained by using [model].get_silhouette_score()""")

        if plot_clusters:
            print('------------------\n')
            print('Clusters visualisation')
            two_most_infl_vars = self._identify_two_influential_vars()
            self.get_bivariate_plots(two_most_infl_vars)
            print("""Only two most influential variables are shown (based on F-statistic's p-value).
To see more plots, use [model].get_bivariate_plots()""")
        
        return self
    
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int 
            Number of digits to round results when showing them
        """
        phrase = 'method {}'
        
        print('\nK-MEANS SUMMARY')
        print('------------------\n')
        if self.is_converged:
            print(f'Estimation was succesfully converged in {self.n_iters} iterations.')
        else:
            print('Estimation was NOT converged successfully.')
            print('Please enlarge the number of iterations.')
        print('------------------\n')
        print('Final cluster centers')
        display(self.get_cluster_centers(round_discrete=True).style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.get_cluster_centers()'))\
                    .set_precision(n_decimals))
        if self._integer_type_vars != []:
            print("""Cluster centers for discrete variables are rounded for better interpretability.
To see the exact centers, use [model].get_cluster_centers(round_discrete=False)""")
        print('------------------\n')
        print('Distances between centers')
        display(self.get_distances_between_centers().style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.get_distances_between_centers()'))\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('ANOVA')
        display(self.get_ANOVA_table().style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.get_ANOVA_table()'))\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('Cluster membership')
        display(self.get_number_of_cases_by_clusters().style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.get_number_of_cases_by_clusters()'))\
                    .set_precision(n_decimals))

    
    def get_cluster_centers(self, round_discrete=False):
        """
        Get the dataframe with cluster centers (means of each variable within clusters).
        
        Parameters
        ----------
        round_discrete : bool 
            Whether to round values of discrete variables for better interpretability

        Returns
        -------
        pd.DataFrame
            A table with cluster centers
        """
        
        centers = self._model.cluster_centers_
        variables = self._variables
        idx = self._cluster_idx
        
        result = pd.DataFrame(
            centers,
            columns=variables,
            index = idx
        )
        
        if round_discrete:
            for var in self._integer_type_vars:
                result[var] = result[var].apply(round)
        
        result.index.name = 'Cluster'
        
        return result
    
    def get_number_of_cases_by_clusters(self):
        """
        Get the frequency table, illustrating the distribution of observations by clusters.
        
        Returns
        -------
        pd.DataFrame
            A table with number of observations in each cluster
        """
        preds = self._predictions.copy()
        dist = pd.DataFrame(
            preds['Cluster'].value_counts(dropna=False).sort_index()
        )
        dist.columns = ['N']
        if len(dist.index) > len(self._cluster_idx):
            dist.index = self._cluster_idx + ['Missing']
        else:
            dist.index = self._cluster_idx
        dist.index.name = 'Cluster'
        dist.loc['Valid'] = dist.loc[self._cluster_idx, 'N'].sum()
        
        return dist.T
    
    def get_ANOVA_table(self):
        """
        Get the results of ANOVA analysis, i.e. analysis of differences in means of each variable between clusters.

        Returns
        -------
        pd.DataFrame
            A table with results of ANOVA test
        """
        preds = self._predictions.copy()
        summary = ANOVA(preds, self._variables, 'Cluster', show_results=False).summary()
        idx_to_show_b = [i for i in summary.index if 'Between Groups' in i]
        idx_to_show_w = [i for i in summary.index if 'Within Groups' in i]
        cols_to_show = ['Mean Square', 'df', 'F', 'p-value']
        summary_b = summary.reindex(idx_to_show_b)[cols_to_show]
        summary_w = summary.reindex(idx_to_show_w)[cols_to_show]
        summary_b.index = self._variables
        summary_w.index = self._variables
        summary = pd.concat([summary_b, summary_w], axis=1)
        summary.columns = [
            'Mean Square (Between Groups)',
            'df (Between Groups)',
            'F',
            'p-value',
            'Mean Square (Within Groups)',
            'df (Within Groups)',
            'F not used',
            'p-value not used']
        columns_to_show = [
            'Mean Square (Between Groups)',
            'df (Between Groups)',
            'Mean Square (Within Groups)',
            'df (Within Groups)',
            'F',
            'p-value']
        return summary[columns_to_show]
    
    def get_distances_between_centers(self):
        """
        Get Eucledian's distance between each cluster's centers.

        Returns
        -------
        pd.DataFrame
            A table with distances between each cluster
        """
        centers = self.cluster_centers
        idx = list(centers.index)
        dists = squareform(pdist(centers))
        dists = pd.DataFrame(dists, index=idx, columns=idx)
        dists.index.name = 'Cluster'
        return dists
        
    def transform(
        self,
        data=None,
        cluster_membership=True,
        distance_to_centers=False,
        add_to_data=False
    ):
        """
        Assign each observation to a relevant cluster and / or estimate distance between each observation 
        and all clusters' centers.
        
        Parameters
        ----------
        data : pd.DataFrame 
            Data for which new variables are estimated.
            If not specified, data that were used to fit a model are used.
        cluster_membership : bool 
            Whether to return cluster membership for each observation
        distance_to_centers : bool 
            Whether to return distance to cluster centers for each observation
        add_to_data : bool 
            Whether to merge new values with the given data

        Returns
        -------
        pd.DataFrame
            Requested transformations
        """        
        if data is None:
            data_copy = self._data.copy()
            transform_data = self._data[self._variables].dropna().copy()
        else:
            data_copy = data.copy()
            transform_data = data[self._variables].dropna().copy()
        
        observation_idx = list(transform_data.index)
        
        cl_memb = self._model.predict(transform_data) + 1
        dist = self._model.transform(transform_data)    
        
        cl_memb = pd.DataFrame(
            cl_memb,
            index=observation_idx,
            columns=['Cluster'],
            dtype=int
        )
        
        dist_cols = [f'Distance to cluster {i}' for i in self._cluster_idx]
        dist = pd.DataFrame(
            dist,
            index=observation_idx,
            columns=dist_cols
        )
        
        columns_to_show = []
        
        if add_to_data:
            columns_to_show.extend(list(data_copy.columns))
        if cluster_membership:
            columns_to_show.append('Cluster')
        if distance_to_centers:
            columns_to_show.extend(dist_cols)      
        
        data_copy['Cluster'] = cl_memb
        data_copy[dist_cols] = dist        

        result = data_copy[columns_to_show].copy()
        
        return result
    
    def get_bivariate_plots(self, variables=None):
        """
        Visualize clusters in a bivariate dimension of all or the requested variables.
        
        Parameters
        ----------
        variables : list
            List of names of variables for which plots should be shown.
            If not specified, all variables that were used to fit a model are used.
        """ 
        if variables is None:
            variables = self._variables
        combos = list(combinations(variables, 2))
        data = self._predictions[variables + ['Cluster']]
        centers = self.cluster_centers[variables]
        for combo in combos:
            KMeans._one_bivariate_cluster_plot(data, combo, centers[list(combo)])
    
    @staticmethod
    def _one_bivariate_cluster_plot(data, two_variables, centers):
        var1, var2 = two_variables
        data = data[list(two_variables) + ['Cluster']].dropna()
        plt.figure(figsize=(8, 5))
        plt.scatter(data[var1], data[var2], c=data['Cluster'], cmap='Set3')
        plt.scatter(centers.loc[:, var1], centers.loc[:, var2], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers.values):
            k = i + 1
            plt.scatter(c[0], c[1], marker='$%d$' % k, alpha=1,
                        s=50, edgecolor='k')
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title('Cluster membership')
        plt.show();
        
    def _identify_two_influential_vars(self):
        anv = self.get_ANOVA_table()
        anv = anv.sort_values('p-value')
        idx_infl = list(anv.index)[:2]
        return idx_infl
    
    def get_silhouette_score(self):
        """
        Return a value of the average silhouette score for the current model.
        
        Returns
        -------
        float
            A silhouette coefficient
        """
        data = self._predictions[self._variables].dropna()
        labels = np.ravel(self._predictions['Cluster'].dropna())
        return silhouette_score(data, labels)
    
    def get_silhouette_plot(self):
        """
        Visualize the model's quality based on its average silhouette score.
        
        """
        silh_score = self.get_silhouette_score()
        silh_point = 1 + silh_score
        plt.figure(figsize=(8, 2))
        plt.barh(0.1, 2, left=-1, height=0.2, color='lightgreen', alpha=0.5)
        plt.barh(0.1, 1.5, left=-1, height=0.2, color='papayawhip', alpha=0.6)
        plt.barh(0.1, 1.2, left=-1, height=0.2, color='salmon', alpha=0.7)
        plt.barh(0.1, silh_point, left=-1, height=0.1, color='orchid', edgecolor='black')

        plt.scatter([silh_score], [0.1], c='black', zorder=2)
        plt.annotate(round(silh_score, 2), (silh_score+0.01, 0.1+0.01))
        plt.annotate('Poor', (0.07, 0.01))
        plt.annotate('Fair', (0.37, 0.01))
        plt.annotate('Good', (0.85, 0.01))
        plt.yticks([])
        plt.title('Solution quality')
        plt.show();