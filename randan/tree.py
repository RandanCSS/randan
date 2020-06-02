from .utils import *
from .bivariate_association import Crosstab
from .comparison_of_central_tendency import ANOVA

import pandas as pd
import numpy as np
import os
import pydot
from IPython.display import Image, display

from sklearn.metrics import r2_score as r2

class CHAIDClassifier:
    """
    Class to build the CHAID tree on a categorical dependent variable.

    Parameters
    ----------
    method : 'pearson' or 'likelihood' 
        Which Chi-square metric to use
        (CURRENTLY ONLY 'PEARSON' IS AVAILABLE)
    max_depth : int 
        Maximum possible depth of tree
    min_child_node : int 
        Minimum possible number of observations in a child node
    min_parent_node : int 
        Minimum possible number of observations in a parent node
    sig_level_split : float 
        Significance level to split a node
    sig_level_merge : float 
        Significance level to merge categories
    bonferroni : bool 
        (CURRENTLY UNAVAILIABLE) Whether to use a Bonferroni's adjustment for p-values
    allow_resplit : bool 
        (CURRENTLY UNAVAILIABLE) Whether to re-split merged categories
    n_intervals : int 
        Maximum possible number of intervals to bin scale variables

    Attributes
    ----------
    depth : int
        The depth of the obtained tree
    nodes : pd.DataFrame
        A table aggregating information about all nodes of a tree
    terminal_nodes : pd.DataFrame
        A table aggregating information about terminal nodes of a tree
    significant_variables : list
        Names of variables remained in a tree
    classification_table : pd.DataFrame
        A classification table for train data
    precision_and_recall : pd.DataFrame
        A table with precision, recall, and f1-score for train data
    classification_table_test : pd.DataFrame
        A classification table for test data
    precision_and_recall_test : pd.DataFrame
        A table with precision, recall, and f1-score for test data
    """
    def __init__(self,
                method='pearson',
                max_depth=3,
                min_child_node=50,
                min_parent_node=100,
                sig_level_split=0.05,
                sig_level_merge=0.05,
                bonferroni=True,
                allow_resplit=False,
                n_intervals=10):
        
        self.method = method
        self.max_depth = max_depth
        self.min_child_node = min_child_node
        self.min_parent_node = min_parent_node
        self.sig_level_split = sig_level_split
        self.sig_level_merge = sig_level_merge
        self.bonferroni = bonferroni
        self.allow_resplit = allow_resplit
        self.n_intervals = n_intervals
        
        self.nodes = []
        
    def fit(self,
            data,
            dependent_variable,
            independent_variables,
            scale_variables=[],
            ordinal_variables=[],
            test_data=None,
            show_results=True,
            plot_tree=True,
            save_plot=False,
            save_plot_path='',
            save_plot_name='tree',
            save_plot_format='png',
            tree_in_table_format=False,
            n_decimals=3):

        """
        Fit a model to the given data.

        Parameters
        ----------
        data : pd.DataFrame 
            Data to fit a model (train data)
        dependent_variable : str 
            Name of a categorical dependent variable
        independent_variables : list 
            List of names of independent variables
        scale_variables : list 
            Names of independent variables that should
            be considered as scale variables
        ordinal_variables : list 
            Names of independent variables that should
            be considered as ordinal variables; to use this option properly 
            you should either add digits to the labels 
            (i.e., 'low', 'medium', 'high' -> '1. low', '2. medium', '3. high'),
            or turn data into a numeric dtype (1, 2, 3),
            or convert your variable to pandas.Categorical and set the order,
            see more: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Categorical.html
        test_data : pd.DataFrame
            Data to test a model
        show_results : bool 
            Whether to show results of analysis
        plot_tree : bool 
            Whether to plot the obtained tree
        save_plot : bool 
            Whether to save a plot
        save_plot_path : str 
            Folder to save a plot
        save_plot_name :str 
            What name to use when save a plot
        save_plot_format : str 
            What format to use when save a plot ('png', 'svg', 'jpg')
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        n_decimals : int 
            Number of digits to round results when showing them

        Returns
        -------
        self
            The current instance of the CHAIDClassifier class
        """
        #todo - handle missing data
        #todo - allow resplit
        #todo - bonferroni
        #todo - different methods in crosstab
        
        self._initial_data = data
        data = data.dropna(subset=independent_variables+[dependent_variable])
        self._data = data.copy()
        self._dependent_variable = dependent_variable
        self._independent_variables = independent_variables
        self._ordinal_variables = ordinal_variables
        self._scale_variables = scale_variables
        self._use_test_data = True if test_data is not None else False
        if self._use_test_data:
            self._test_data = test_data.dropna(subset=independent_variables+[dependent_variable])
        
        self._mode = self._data[dependent_variable].mode().item()

        if self.max_depth is None:
            self.max_depth = len(independent_variables)
        
        tree_nodes = []
        
        tree_nodes.extend(self._build_nodes_from_variable(data,
                                                          dependent_variable))
        
        
        
        tree_nodes = self._split_nodes(data,
                                       dependent_variable,
                                       independent_variables,
                                       tree_nodes)
                                     
        self.nodes = pd.DataFrame(tree_nodes)
        
        
        if len(self.nodes) > 1:
            terminal_nodes_idx = self.nodes.apply(lambda x: self._check_if_terminal_node(x), axis=1)
            self.terminal_nodes = self.nodes[terminal_nodes_idx]
            terminal_nodes_lst = [tree_nodes[i] for i in self.terminal_nodes.index]
            depths = [CHAIDClassifier._get_node_depth(node, tree_nodes)\
                                for node in terminal_nodes_lst]
            self.depth = max(depths)
            
            self._node_interactions = self.get_interactions() 
            self.significant_variables = self.get_significant_variables()
        else:
            self.terminal_nodes = []
            self.depth = 0
            self.significant_variables = []
            self._data[f'{dependent_variable} (predicted)'] = self._data[dependent_variable].mode().item()
        
        self.classification_table = self.get_classification_table()
        self.precision_and_recall = self.get_precision_and_recall()
        
        if self._use_test_data:
            test_predictions = self.predict(self._test_data)
            self.classification_table_test = classification_table(self._test_data[dependent_variable],
            test_predictions[f'{dependent_variable} (predicted)'])
            self.precision_and_recall_test = precision_and_recall(self.classification_table_test)

        if show_results:
            self.show_results(tree_in_table_format=tree_in_table_format, 
                              n_decimals=n_decimals)
        if plot_tree:
            print('------------------\n')
            print('Tree plot')
            self.plot_tree(
                save_plot=save_plot,
                save_plot_path=save_plot_path,
                save_plot_name=save_plot_name,
                save_plot_format=save_plot_format)
        
        return self
    
    def get_significant_variables(self):
        """
        Identify which variables remained in the tree,
        i.e. should be considered as significant ones.

        Returns
        -------
        list
            Names of significant variables
        """

        significant_variables = get_categories(self.nodes['Variable'])
        significant_variables.remove(self._dependent_variable)  
        return significant_variables
    
    def plot_tree(self,
            save_plot=False,
            save_plot_path=None,
            save_plot_name='tree',
            save_plot_format='png'):
        """
        Plot the obtained tree.

        Parameters
        ----------
        save_plot : bool 
            Whether to save a plot
        save_plot_path : str 
            Folder to save a plot
        save_plot_name : str 
            What name to use when save a plot
        save_plot_format : str 
            What format to use when save a plot ('png', 'svg', 'jpg')
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        """
        
        graph = pydot.Dot(graph_type='graph', rankdir='LR')

        node_labels = {}

        dep_var = self._dependent_variable
        categories = list(self.nodes.loc[0, 'Dependent variable'].keys())
        colors = dict(zip(categories, available_colors[:len(categories)]))

        #chi2s = []

        for idx, node in self.nodes.iterrows():

            current_node = node['Node']
            node_variable = node['Variable']
            if 'Category' in node:
                node_category = node['Category']
            predicted_category = node['Mode']
            n = node['N']
            if 'Parent node' in node:
                parent_node = node['Parent node']
            else:
                parent_node = None
            if 'Chi-square' in node:
                chi2 = round(node['Chi-square'], 1)
            fillcolor=colors[predicted_category]
            if current_node == 'Node 0':
                node_label = {
                    current_node: f'{current_node}\n{node_variable}\nN = {n}\nMode = {predicted_category}'
                }

            else:
                node_label = {
                    current_node: f'{current_node}\n{node_variable} =\n{node_category}\nN = {n}\n{dep_var} = {predicted_category}'
                }

            node_labels.update(node_label)    

            node_dot = pydot.Node(node_labels[current_node],
                                  style='"rounded,filled"',
                                  fillcolor=fillcolor,
                                  shape='box')
            graph.add_node(node_dot)

            if pd.notnull(parent_node):
                graph.add_edge(pydot.Edge(node_labels[parent_node], node_labels[current_node]))
        if save_plot:
            if save_plot_path is not None:
                path = os.path.join(save_plot_path, f'{save_plot_name}.{save_plot_format}')
            else: 
                path = f'{save_plot_name}.{save_plot_format}'
                
            if save_plot_format.lower().strip()=='png':
                graph.write_png(path, encoding='utf-8')
            elif save_plot_format.lower().strip() in ['jpg', 'jpeg']:
                graph.write_jpg(path, encoding='utf-8')
            elif save_plot_format.lower().strip()=='svg':
                graph.write_svg(path, encoding='utf-8')
            else:
                raise ValueError('Unknown image format. Available options: png, jpg, svg')
                
        display(Image(graph.create_png(encoding='utf-8')))
    
    def get_precision_and_recall(self):
        """
        Estimate precision, recall, and F-score for all the categories.

        Returns
        -------
        pd.DataFrame
            A table with estimated metrics
        """

        results = precision_and_recall(self.classification_table)
        return results
    
    def summary(self):
        """
        Get model summary.

        Returns
        -------
        pd.DataFrame
            A summary table
        """
        statistics = ['Growing Method',
                     'Dependent variable',
                     'Independent variables',
                     'Maximum tree depth',
                     'Minimum cases in child node',
                     'Minimum cases in parent node',
                     'Significant variables',
                     'Number of nodes',
                     'Number of terminal nodes',
                     'Depth']
        
        data = ['CHAID',
                f'{self._dependent_variable}',
                f'{", ".join(self._independent_variables)}',
                f'{self.max_depth}',
                f'{self.min_child_node}',
                f'{self.min_parent_node}',
                f'{", ".join(self.significant_variables)}',
                f'{len(self.nodes)-1}',
                f'{len(self.terminal_nodes)}',
                f'{self.depth}']
        
        results = pd.DataFrame(data, index=statistics, columns=[''])
        
        return results
    
    def show_results(self,
                     tree_in_table_format=False,
                     n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        n_decimals : int 
            Number of digits to round results when showing them
        """
        print('\nCHAID SUMMARY')
        print('------------------\n')
        print('Tree information')
        display(self.summary())
        if tree_in_table_format:
            print('------------------\n')
            print('Tree nodes')
            display(self.nodes.style\
                    .format(None, na_rep="")\
                    .set_caption(f'Dependent variable: {self._dependent_variable}, independent variables: {", ".join(self._independent_variables)}')\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('Classification table')
        display(self.classification_table.style\
                .set_precision(1))
        if self._use_test_data:
            print('Classification table for test data')
            display(self.classification_table_test.style\
                .set_precision(1))
        print('------------------\n')
        print('Prediction quality metrics')
        display(self.precision_and_recall.style\
                .set_precision(n_decimals))
        if self._use_test_data:
            print('Prediction quality metrics for test data')
            display(self.precision_and_recall_test.style\
                .set_precision(n_decimals))
                             
    
    def get_classification_table(self):
        """
        Get the classification table as it is shown in SPSS.

        Returns
        -------
        pd.DataFrame
            A classification table
        """
        dependent_variable = self._dependent_variable
        classification = classification_table(self._data[dependent_variable],
        self._data[f'{dependent_variable} (predicted)']
        )
        return classification
        
    def _check_if_terminal_node(self, node):
        if node['Node'] in self.nodes['Parent node'].tolist() or pd.isna(node['Parent node']):
            return False
        else:
            return True
    
    @staticmethod
    def _predict_one_observation(observation,
                                 nodes,
                                 scale_variables,
                                 constant):
        if observation.hasnans:
            return None, None

        predicted_node = 'Node 0'
        while True:
            current_nodes_level = nodes[nodes['Parent node']==predicted_node]
            if len(current_nodes_level) == 0:
                return predicted_category, predicted_node
            split_variable = current_nodes_level['Variable'].iloc[0]
            observation_category = observation[split_variable]
            if split_variable not in scale_variables:
                try:
                    rule = current_nodes_level.loc[:, 'Category'].str.split(' / ').apply(lambda x: True if str(observation_category) in x else False)
                except AttributeError:
                    rule = current_nodes_level.loc[:, 'Category'].astype(float).astype(str).str.split(' / ').apply(lambda x: True if str(float(observation_category)) in x else False)
            else:
                rule = current_nodes_level.loc[:, 'Category'].apply(lambda x: x.overlaps(pd.Interval(observation_category, observation_category, 'both')))
            
            try:
                predicted_category = current_nodes_level[rule]['Mode'].iloc[0]
                predicted_node = current_nodes_level[rule]['Node'].iloc[0]
            except IndexError:
                predicted_category = constant
                predicted_node = 'Out of tree'
    
    def predict(self,
                data=None,
                dependent_variable=True,
                node=False,
                interaction=False,
                add_to_data=False):
        """
        Predict a value of the dependent variable and/or a node for the given data.

        Parameters
        ----------
        data : pd.DataFrame 
            Data for which values or nodes should be predicted
        dependent_variable : bool 
            Whether to predict the value of the dependent variable
        node : bool 
            Whether to predict the node
        interaction : bool 
            Whether to predict interaction corresponded to the node
        add_to_data : bool 
            Whether to merge predictions with the given data

        Returns
        -------
        pd.DataFrame
            Requested values
        """

        if data is None or data[self._independent_variables]\
        .equals(self._initial_data[self._independent_variables]):
            result = self._data[[f'{self._dependent_variable} (predicted)', 'Node']].copy()
        
        else:
        
            data_for_prediction = data[self.significant_variables].copy()
            result = data_for_prediction.apply(lambda x: CHAIDClassifier._predict_one_observation(
                x, 
                self.nodes, 
                self._scale_variables,
                self._mode
            ), axis=1, result_type='expand')
            result.columns = [f'{self._dependent_variable} (predicted)', 'Node']
                                 
        result['Interaction'] = result['Node'].map(self._node_interactions)

        columns_to_show = []                     
        if dependent_variable:
            columns_to_show.append(f'{self._dependent_variable} (predicted)')
        if node:
            columns_to_show.append('Node')
        if interaction:
            columns_to_show.append('Interaction')

        result = result[columns_to_show].copy()
            
        if add_to_data:
            return pd.concat([data, result], axis=1)
        else:
            return result       
    
    
    @staticmethod
    def _get_most_significant_variable(data, 
                                       dependent_variable,
                                       independent_variables,
                                       method):
        #todo - different methods in crosstab (pearson+likelihood)
        results = []
        for variable in independent_variables:
            if len(data[variable].value_counts()) > 1:
                ctab = Crosstab(data, variable, dependent_variable, show_results=False, only_stats=True)
                results.append([ctab.pvalue, ctab.chi_square, ctab.dof])
            else:
                results.append([1, 0, 0])

        if len(results) == 0:
            return None, None, None, None
        else:
            results = pd.DataFrame(results,
                                   columns=['pvalue', 'chi2', 'dof'],
                                   index=independent_variables)
            most_significant_variable = results['pvalue'].idxmin()
            its_pvalue = results.loc[most_significant_variable, 'pvalue']
            its_chi2 = results.loc[most_significant_variable, 'chi2']
            its_dof = results.loc[most_significant_variable, 'dof']
            return most_significant_variable, its_pvalue, its_chi2, its_dof

    def _merging(self,
                 data,
                 dependent_variable,
                 independent_variable):

        #todo - different methods in crosstab (pearson+likelihood)
        
        sig_level = self.sig_level_merge
        method = self.method
        min_child_node = self.min_child_node
        min_parent_node = self.min_parent_node
        
        # ! todo - resplit categories !
        resplit = self.allow_resplit
        
        data = data.copy()
        
        max_pvalue = 1
        
        while max_pvalue > sig_level:           
            categories = get_categories(data[independent_variable])
            
            if len(categories) <= 2:
                if len(categories) == 2 and min_child_node is not None:
                    min_count = data[independent_variable].value_counts().min()
                    if min_count < min_child_node:
                        if independent_variable not in self._scale_variables:
                            data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                               categories[0],
                                                                                               categories[1])
                        else:
                            data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                              categories[0],
                                                                                              categories[1])
                
                return data[independent_variable]
                 
            if independent_variable not in self._ordinal_variables + self._scale_variables:
                categories_combo = get_unordered_combinations(categories, 2)
            else:
                categories_combo = get_ordered_combinations(categories, 2)
            
            results = pd.DataFrame(columns=categories,
                                   index=categories)
            for pair in categories_combo:
                pair_data = data[data[independent_variable].isin(pair)]
                if len(pair_data) > 0:
                    ctab = Crosstab(pair_data, independent_variable, dependent_variable, only_stats=True) #add method
                    var1, var2 = pair
                    results.loc[var1, var2] = ctab.pvalue
                    results.loc[var2, var1] = ctab.pvalue
            max_pvalue = results.max().max()
            if max_pvalue > sig_level:
                cat1 = results.max().idxmax()
                cat2 = results[cat1][results[cat1]==max_pvalue].index[0]
                if independent_variable not in self._scale_variables:
                    data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                       cat1,
                                                                                       cat2)
                else:
                    data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                      cat1,
                                                                                      cat2)
        
        if min_child_node is not None:
            min_count = data[independent_variable].value_counts().min()
            n_categories = len(data[independent_variable].unique())
    #         (Optional) Any category having too few observations (as compared with a user-specified minimum segment size) 
    #         is merged with the most similar other category as measured by the largest of the p-values.

            while 0 < min_count < min_child_node and n_categories > 1:                

                min_count_category = data[independent_variable].value_counts().idxmin()
                categories = get_categories(data[independent_variable])
                if independent_variable not in self._ordinal_variables + self._scale_variables:
                    categories_combo = get_unordered_combinations(categories, 2)
                else:
                    categories_combo = get_ordered_combinations(categories, 2)
                results = pd.DataFrame(columns=categories,
                       index=categories)
                for pair in categories_combo:
                    pair_data = data[data[independent_variable].isin(pair)]
                    if len(pair_data) > 0:
                        ctab = Crosstab(pair_data, independent_variable, dependent_variable, only_stats=True) #add method
                        var1, var2 = pair
                        results.loc[var1, var2] = ctab.pvalue
                        results.loc[var2, var1] = ctab.pvalue

                max_pvalue = results[min_count_category].max()
                max_pvalue_category = pd.to_numeric(results[min_count_category]).idxmax()
                if independent_variable not in self._scale_variables:
                    data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                 min_count_category,
                                                                                 max_pvalue_category)
                else:
                    data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                 min_count_category,
                                                                                 max_pvalue_category)          
                min_count = data[independent_variable].value_counts().min()
                n_categories = len(data[independent_variable].unique())
        return data[independent_variable]

    def _build_nodes_from_variable(self,
                                   data,
                                   dependent_variable,
                                   independent_variable=None,
                                   start_counter=0,
                                   parent_node=None):
        
        if independent_variable is not None:
            categories = get_categories(data[independent_variable])
            nodes = []
            counter = start_counter
            n_observations = []
            for category in categories:
                filter_data = data[data[independent_variable]==category][dependent_variable]
                n = len(filter_data)
                n_observations.append(n)
            small_n = [n for n in n_observations if n <= self.min_child_node]   
            if len(small_n) < len(categories) - 1:
                for category in categories:
                    filter_data = data[data[independent_variable]==category][dependent_variable]
                    n = len(filter_data)
                    dependent_variable_dist = dict((round(filter_data.value_counts(normalize=True, sort=False)*100, 3).items()))
                    mode = filter_data.value_counts().idxmax()
                    node_info = {'Node': f'Node {counter}',
                                 'Parent node': parent_node,
                                 'Variable': independent_variable,
                                 'Category': category,
                                 'Dependent variable': dependent_variable_dist,
                                 'Mode': mode,
                                 'N': n}
                    nodes.append(node_info)
                    counter += 1
        else:
            filter_data = data[dependent_variable]
            n = len(filter_data)
            dependent_variable_dist = dict((round(filter_data.value_counts(normalize=True, sort=False)*100, 3).items()))
            mode = filter_data.value_counts().idxmax()
            node_info = {'Node': f'Node 0',
                         'Variable': dependent_variable,
                         'Dependent variable': dependent_variable_dist,
                         'Mode': mode,
                         'N': n}
            nodes = [node_info]                                
                
        return nodes
    
    def _split_nodes(self,
                     data,
                     dependent_variable,
                     independent_variables,
                     tree_nodes,
                     parent_node='Node 0'):
        
        data_iter = data.copy()
        for variable in independent_variables:
            
            if variable in self._scale_variables:
                
                data_iter[variable] = binning(data_iter[variable], self.n_intervals)
            
            data_iter[variable] = self._merging(data_iter,
                                                dependent_variable,
                                                variable)
        
        split_variable, split_pvalue, split_chi2, split_dof = CHAIDClassifier\
        ._get_most_significant_variable(data_iter,
                                        dependent_variable,
                                        independent_variables,
                                        self.method)
        
        if split_variable is not None:
            split_info = {
                         'Chi-square': split_chi2,
                         'p-value': split_pvalue,
                         'dof': split_dof}
            if split_info['p-value'] <= self.sig_level_split:
                iter_nodes = self._build_nodes_from_variable(data_iter,
                                                                       dependent_variable,
                                                                       split_variable,
                                                                       start_counter=len(tree_nodes),
                                                                       parent_node=parent_node)
                for node in iter_nodes:
                    node.update(split_info)
                
                if len(iter_nodes) >= 2:
                    tree_nodes.extend(iter_nodes)
                    current_depths = [CHAIDClassifier._get_node_depth(node, tree_nodes)\
                                        for node in iter_nodes if node['Node']!='Node 0']
                    current_depth = max(current_depths)
                    if current_depth < self.max_depth:    
                        
                        for i in range(len(iter_nodes)):
                            parent_node = iter_nodes[i]['Node']
                            node_variable = iter_nodes[i]['Variable']
                            node_category = iter_nodes[i]['Category']
                            node_prediction = iter_nodes[i]['Mode']
                            data_per_node = data[data_iter[node_variable]==node_category].copy()
                            self._data.loc[data_per_node.index, f'{self._dependent_variable} (predicted)'] = node_prediction
                            self._data.loc[data_per_node.index, 'Node'] = parent_node
                            if len(data_per_node) >= self.min_parent_node:
                                available_variables = independent_variables.copy()
                                available_variables.remove(node_variable)
                                tree_nodes = self._split_nodes(data_per_node,
                                                                   dependent_variable,
                                                                   available_variables,
                                                                   tree_nodes,
                                                                   parent_node)
                    else:
                        for i in range(len(iter_nodes)):
                            parent_node = iter_nodes[i]['Node']
                            node_variable = iter_nodes[i]['Variable']
                            node_category = iter_nodes[i]['Category']
                            node_prediction = iter_nodes[i]['Mode']
                            data_per_node = data[data_iter[node_variable]==node_category].copy()
                            self._data.loc[data_per_node.index, f'{self._dependent_variable} (predicted)'] = node_prediction
                            self._data.loc[data_per_node.index, 'Node'] = parent_node
        

        return tree_nodes
    
    #move this method to the base chaid tree class
    @staticmethod
    def _get_node_depth(node, nodes):
        depth = 0
        while True:
            if 'Parent node' in node:
                parent_node = node['Parent node']
                node = nodes[int(parent_node.split()[-1])]
                depth += 1
            else:
                break
        return depth
            
    @staticmethod
    def _get_one_node_interaction(node, nodes):
        interaction = []
        if node['Node'] == 'Node 0':
            return np.nan
        while True:
            if node['Node'] == 'Node 0':
                break
            interaction.append(f'{node["Variable"]} = {node["Category"]}')
            node = nodes[nodes['Node']==node['Parent node']].iloc[0]
        return ' * '.join(interaction)
                               
    def get_interactions(self, result='dict'):
        """
        Return a dictionary or a DataFrame with nodes and interactions corresponded to them.

        Parameters
        ----------
        result : 'dict' or 'DataFrame'
            Data type of results

        Returns
        -------
        dict or pd.DataFrame
            Definitions of nodes in terms of interactions
        """
                               
        results = self.nodes.apply(lambda x: CHAIDClassifier._get_one_node_interaction(x, self.nodes), axis=1)
        results.index = self.nodes['Node']
        if result.lower() == 'dict':
            return dict(results)
        elif result.lower() == 'dataframe':
            return pd.DataFrame(results, columns=['Interaction'])
        else:
            raise ValueError("Unknown result type. Possible values: 'dict' and 'DataFrame'.")


class CHAIDRegressor:
    """
    Class to build the CHAID tree on a scale dependent variable.

    Parameters
    ----------
    max_depth : int 
        Maximum possible depth of tree
    min_child_node : int 
        Minimum possible number of observations in a child node
    min_parent_node : int 
        Minimum possible number of observations in a parent node
    sig_level_split : float 
        Significance level to split a node
    sig_level_merge : float 
        Significance level to merge categories
    bonferroni : bool 
        (CURRENTLY UNAVAILIABLE) Whether to use a Bonferroni's adjustment for p-values
    allow_resplit : bool 
        (CURRENTLY UNAVAILIABLE) Whether to re-split merged categories
    n_intervals : int 
        Maximum possible number of intervals to bin scale variables

    Attributes
    ----------
    depth : int
        The depth of the obtained tree
    nodes : pd.DataFrame
        A table aggregating information about all nodes of a tree
    terminal_nodes : pd.DataFrame
        A table aggregating information about terminal nodes of a tree
    significant_variables : list
        Names of variables remained in a tree
    r2 : float
        R2 score on train data
    r2_test : float 
        R2 score on test data
    """
    def __init__(self,
                 max_depth=3,
                min_child_node=50,
                min_parent_node=100,
                sig_level_split=0.05,
                sig_level_merge=0.05,
                bonferroni=True,
                allow_resplit=False,
                n_intervals=10):
        
        self.max_depth = max_depth
        self.min_child_node = min_child_node
        self.min_parent_node = min_parent_node
        self.sig_level_split = sig_level_split
        self.sig_level_merge = sig_level_merge
        self.bonferroni = bonferroni
        self.allow_resplit = allow_resplit
        self.n_intervals = n_intervals
        self.nodes = []
        
    def fit(self,
            data,
            dependent_variable,
            independent_variables,
            scale_variables=[],
            ordinal_variables=[],
            test_data=None,
            show_results=True,
            plot_tree=True,
            save_plot=False,
            save_plot_path='',
            save_plot_name='tree',
            save_plot_format='png',
            tree_in_table_format=False,
            n_decimals=3):
        """
        Fit model to the given data.

        Parameters
        ----------
        data : pd.DataFrame 
            Data to fit a model (train data)
        dependent_variable : str 
            Name of a categorical dependent variable
        independent_variables : list 
            List of names of independent variables
        scale_variables : list 
            Names of independent variables that should
            be considered as scale variables
        ordinal_variables : list 
            Names of independent variables that should
            be considered as ordinal variables; to use this option properly 
            you should either add digits to the labels 
            (i.e., 'low', 'medium', 'high' -> '1. low', '2. medium', '3. high'),
            or turn data into a numeric dtype (1, 2, 3),
            or convert your variable to pandas.Categorical and set the order,
            see more: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Categorical.html
        test_data : pd.DataFrame
            Data to test a model
        show_results : bool 
            Whether to show results of analysis
        plot_tree : bool 
            Whether to plot the obtained tree
        save_plot : bool 
            Whether to save a plot
        save_plot_path : str 
            Folder to save a plot
        save_plot_name :str 
            What name to use when save a plot
        save_plot_format : str 
            What format to use when save a plot ('png', 'svg', 'jpg')
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        n_decimals : int 
            Number of digits to round results when showing them

        Returns
        -------
        self
            The current instance of the CHAIDClassifier class
        """
        #todo - handle missing data
        #todo - allow resplit
        #todo - bonferroni
        #todo - different methods in crosstab
        
        self._initial_data = data
        data = data.dropna(subset=independent_variables+[dependent_variable])
        self._data = data.copy()
        self._dependent_variable = dependent_variable
        self._independent_variables = independent_variables
        self._ordinal_variables = ordinal_variables
        self._scale_variables = scale_variables
        self._use_test_data = True if test_data is not None else False
        if self._use_test_data:
            self._test_data = test_data.dropna(subset=independent_variables+[dependent_variable])

        self._mean = self._data[dependent_variable].mean()

        if self.max_depth is None:
            self.max_depth = len(independent_variables)
        
        tree_nodes = []
        
        tree_nodes.extend(self._build_nodes_from_variable(data,
                                                          dependent_variable))
        #print(tree_nodes)
        
        
        tree_nodes = self._split_nodes(data,
                                       dependent_variable,
                                       independent_variables,
                                       tree_nodes)
                                     

        self.nodes = pd.DataFrame(tree_nodes)

        if len(self.nodes) > 1:
            terminal_nodes_idx = self.nodes.apply(lambda x: self._check_if_terminal_node(x), axis=1)
            self.terminal_nodes = self.nodes[terminal_nodes_idx]
            terminal_nodes_lst = [tree_nodes[i] for i in self.terminal_nodes.index]
            depths = [CHAIDRegressor._get_node_depth(node, tree_nodes)\
                                for node in terminal_nodes_lst]
            self.depth = max(depths)

            self._node_interactions = self.get_interactions()      
            self.significant_variables = self.get_significant_variables()
        else:
            self.terminal_nodes = []
            self.depth = 0
            self.significant_variables = []
            self._data[f'{dependent_variable} (predicted)'] = self._data[dependent_variable].mean()
        
        self.r2 = r2(self._data[dependent_variable],
                    self._data[f'{dependent_variable} (predicted)'])
        
        if self._use_test_data:
            test_predictions = self.predict(self._test_data)
            self.r2_test = r2(self._test_data[dependent_variable],
                    test_predictions[f'{dependent_variable} (predicted)'])

        if show_results:
            self.show_results(tree_in_table_format=tree_in_table_format, 
                              n_decimals=n_decimals)
        if plot_tree:
            print('------------------\n')
            print('Tree plot')
            self.plot_tree(
                save_plot=save_plot,
                save_plot_path=save_plot_path,
                save_plot_name=save_plot_name,
                save_plot_format=save_plot_format)
        
        return self
    
    def get_significant_variables(self):
        """
        Identify which variables remained in the tree,
        i.e. should be considered as significant ones.

        Returns
        -------
        list
            Names of significant variables
        """
        significant_variables = get_categories(self.nodes['Variable'])
        significant_variables.remove(self._dependent_variable)  
        return significant_variables
        
    def plot_tree(self,
            save_plot=False,
            save_plot_path=None,
            save_plot_name='tree',
            save_plot_format='png'):
        """
        Plot the obtained tree.

        Parameters
        ----------
        save_plot : bool 
            Whether to save a plot
        save_plot_path : str 
            Folder to save a plot
        save_plot_name : str 
            What name to use when save a plot
        save_plot_format : str 
            What format to use when save a plot ('png', 'svg', 'jpg')
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        """
        
        graph = pydot.Dot(graph_type='graph', rankdir='LR')

        node_labels = {}

        dep_var = self._dependent_variable

        for idx, node in self.nodes.iterrows():

            current_node = node['Node']
            node_variable = node['Variable']
            node_category = node['Category']
            predicted_category = node['Mean']
            std = node['St. dev.']
            n = node['N']
            parent_node = node['Parent node']
            
            if current_node == 'Node 0':
                node_0_ci = confidence_interval_mean(predicted_category, std, n, sig_level=self.sig_level_split)
                predicted_category = round(predicted_category, 2)
                std = round(std, 2)
                fillcolor='lightgray'
                #fillcolor=colors[predicted_category]
                node_label = {
                    current_node: f'{current_node}\n{node_variable}\nN = {n}\nMean = {predicted_category}\n(std = {std})'
                }

            else:
                node_x_ci = confidence_interval_mean(predicted_category, std, n, sig_level=self.sig_level_split)
                if confidence_interval_comparison(node_x_ci, node_0_ci)=='overlap':                    
                    fillcolor = 'lightgray'
                elif confidence_interval_comparison(node_x_ci, node_0_ci)=='lower':
                    fillcolor = available_colors[-2]
                elif confidence_interval_comparison(node_x_ci, node_0_ci)=='bigger':
                    fillcolor = available_colors[-1]
                    
                predicted_category = round(predicted_category, 2)
                std = round(std, 2)
                node_label = {
                    current_node: f'{current_node}\n{node_variable} =\n{node_category}\nN = {n}\n{dep_var} = {predicted_category}\n(std = {std})'
                }

            node_labels.update(node_label)    

            node_dot = pydot.Node(node_labels[current_node],
                                  style='"rounded,filled"',
                                  fillcolor=fillcolor,
                                  shape='box')
            graph.add_node(node_dot)

            if pd.notnull(parent_node):
                graph.add_edge(pydot.Edge(node_labels[parent_node], node_labels[current_node]))
        if save_plot:
            if save_plot_path is not None:
                path = os.path.join(save_plot_path, f'{save_plot_name}.{save_plot_format}')
            else: 
                path = f'{save_plot_name}.{save_plot_format}'
                
            if save_plot_format.lower().strip()=='png':
                graph.write_png(path, encoding='utf-8')
            elif save_plot_format.lower().strip() in ['jpg', 'jpeg']:
                graph.write_jpg(path, encoding='utf-8')
            elif save_plot_format.lower().strip()=='svg':
                graph.write_svg(path, encoding='utf-8')
            else:
                raise ValueError('Unknown image format. Available options: png, jpg, svg')
                
        display(Image(graph.create_png(encoding='utf-8')))
    
    
    def summary(self):
        """
        Get model summary.

        Returns
        -------
        pd.DataFrame
            A summary table
        """
        statistics = ['Growing Method',
                     'Dependent variable',
                     'Independent variables',
                     'Maximum tree depth',
                     'Minimum cases in child node',
                     'Minimum cases in parent node',
                     'Significant variables',
                     'Number of nodes',
                     'Number of terminal nodes',
                     'Depth',
                     '',
                     'R2']
        
        data = ['CHAID',
                f'{self._dependent_variable}',
                f'{", ".join(self._independent_variables)}',
                f'{self.max_depth}',
                f'{self.min_child_node}',
                f'{self.min_parent_node}',
                f'{", ".join(self.significant_variables)}',
                f'{len(self.nodes)-1}',
                f'{len(self.terminal_nodes)}',
                f'{self.depth}',
               '',
               f'{round(self.r2, 3)}']
        
        results = pd.DataFrame(data, index=statistics, columns=[''])
        
        if self._use_test_data:
            results.loc['R2 (test data)'] = round(self.r2_test, 3)

        return results
    
    def show_results(self,
                     tree_in_table_format=False,
                     n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        tree_in_table_format : bool 
            Whether to show tree as a table when showing results
        n_decimals : int 
            Number of digits to round results when showing them
        """
        
        print('\nCHAID SUMMARY')
        print('------------------\n')
        print('Tree information')
        display(self.summary())
        if tree_in_table_format:
            print('------------------\n')
            print('Tree nodes')
            display(self.nodes.style\
                    .format(None, na_rep="")\
                    .set_caption(f'Dependent variable: {self._dependent_variable}, independent variables: {", ".join(self._independent_variables)}')\
                    .set_precision(n_decimals))
                                 
    def _check_if_terminal_node(self, node):
        if node['Node'] in self.nodes['Parent node'].tolist() or pd.isna(node['Parent node']):
            return False
        else:
            return True
    
    @staticmethod
    def _predict_one_observation(observation,
                                 nodes,
                                 scale_variables,
                                 constant):
        if observation.hasnans:
            return None, None

        predicted_node = 'Node 0'
        
        while True:
            current_nodes_level = nodes[nodes['Parent node']==predicted_node]
            #display(current_nodes_level)
            if len(current_nodes_level) == 0:
                return predicted_category, predicted_node
            split_variable = current_nodes_level['Variable'].iloc[0]
            #print(split_variable)
            observation_category = observation[split_variable]
            #print(observation_category)
            if split_variable not in scale_variables:
                try:
                    rule = current_nodes_level.loc[:, 'Category'].str.split(' / ').apply(lambda x: True if str(observation_category) in x else False)
                except AttributeError:
                    rule = current_nodes_level.loc[:, 'Category'].astype(float).astype(str).str.split(' / ').apply(lambda x: True if str(float(observation_category)) in x else False)
            else:
                rule = current_nodes_level.loc[:, 'Category'].apply(lambda x: x.overlaps(pd.Interval(observation_category, observation_category, 'both')))
            try:
                predicted_category = current_nodes_level[rule]['Mean'].iloc[0]
                #print(predicted_category)
                predicted_node = current_nodes_level[rule]['Node'].iloc[0]
            except IndexError:
                predicted_category = constant
                predicted_node = 'Out of tree'
            #print(predicted_node)
    
    def predict(self,
                data=None,
                dependent_variable=True,
                node=False,
                interaction=False,
                add_to_data=False):
        """
        Predict a value of the dependent variable and/or a node for the given data.

        Parameters
        ----------
        data : pd.DataFrame 
            Data for which values or nodes should be predicted
        dependent_variable : bool 
            Whether to predict the value of the dependent variable
        node : bool 
            Whether to predict the node
        interaction : bool 
            Whether to predict interaction corresponded to the node
        add_to_data : bool 
            Whether to merge predictions with the given data

        Returns
        -------
        pd.DataFrame
            Requested values
        """
        
        if data is None:
            result = self._data[[f'{self._dependent_variable} (predicted)', 'Node']].copy()
        
        else:
        
            data_for_prediction = data[self.significant_variables]
            result = data_for_prediction.apply(lambda x: CHAIDRegressor._predict_one_observation(
                x, 
                self.nodes, 
                self._scale_variables,
                self._mean
            ), axis=1, result_type='expand')
            result.columns = [f'{self._dependent_variable} (predicted)', 'Node']
                                 
        result['Interaction'] = result['Node'].map(self._node_interactions)

        columns_to_show = []                     
        if dependent_variable:
            columns_to_show.append(f'{self._dependent_variable} (predicted)')
        if node:
            columns_to_show.append('Node')
        if interaction:
            columns_to_show.append('Interaction')

        result = result[columns_to_show]
            
        if add_to_data:
            return pd.concat([data, result], axis=1)
        else:
            return result       
    
    @staticmethod
    def _get_most_significant_variable(data, 
                                       dependent_variable,
                                       independent_variables):
        results = []
        for variable in independent_variables:
            if len(data[variable].value_counts()) > 1:
                anv = ANOVA(data, dependent_variable, variable, show_results=False)
                results.append([anv.pvalue, anv.F, anv.dof_b, anv.dof_w])
            else:
                results.append([1, 0, 0, 0])

        if len(results) == 0:
            return None, None, None, None, None
        else:
            results = pd.DataFrame(results,
                                   columns=['pvalue', 'F', 'dof_b', 'dof_w'],
                                   index=independent_variables)

            most_significant_variable = results['pvalue'].idxmin()
            its_pvalue = results.loc[most_significant_variable, 'pvalue']
            its_chi2 = results.loc[most_significant_variable, 'F']
            its_dof_b = results.loc[most_significant_variable, 'dof_b']
            its_dof_w = results.loc[most_significant_variable, 'dof_w']

            return most_significant_variable, its_pvalue, its_chi2, its_dof_b, its_dof_w

        
    def _merging(self,
                 data,
                 dependent_variable,
                 independent_variable):
        
        sig_level = self.sig_level_merge
        min_child_node = self.min_child_node
        min_parent_node = self.min_parent_node
        
        # ! todo - resplit categories !
        resplit = self.allow_resplit
        
        data = data.copy()
        
        max_pvalue = 1
        
        while max_pvalue > sig_level:            
            categories = get_categories(data[independent_variable])
            
            if len(categories) <= 2:
                if len(categories) == 2 and min_child_node is not None:
                    min_count = data[independent_variable].value_counts().min()
                    if min_count < min_child_node:
                        if independent_variable not in self._scale_variables:
                            data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                               categories[0],
                                                                                               categories[1])
                        else:
                            data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                              categories[0],
                                                                                              categories[1])
                return data[independent_variable]
                 
            if independent_variable not in self._ordinal_variables + self._scale_variables:
                categories_combo = get_unordered_combinations(categories, 2)
            else:
                categories_combo = get_ordered_combinations(categories, 2)
            
            results = pd.DataFrame(columns=categories,
                                   index=categories)
            for pair in categories_combo:
                pair_data = data[data[independent_variable].isin(pair)]
                if len(pair_data) > 0:
                    anv = ANOVA(pair_data, dependent_variable, independent_variable, show_results=False)
                    var1, var2 = pair
                    results.loc[var1, var2] = anv.pvalue
                    results.loc[var2, var1] = anv.pvalue
            max_pvalue = results.max().max()
            if max_pvalue > sig_level:
                cat1 = results.max().idxmax()
                cat2 = results[cat1][results[cat1]==max_pvalue].index[0]
                if independent_variable not in self._scale_variables:
                    data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                       cat1,
                                                                                       cat2)
                else:
                    data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                      cat1,
                                                                                      cat2)
            
        if min_child_node is not None:
            min_count = data[independent_variable].value_counts().min()
            n_categories = len(data[independent_variable].unique())
    #         (Optional) Any category having too few observations (as compared with a user-specified minimum segment size) 
    #         is merged with the most similar other category as measured by the largest of the p-values.

            while 0 < min_count < min_child_node and n_categories > 1:                

                min_count_category = data[independent_variable].value_counts().idxmin()
                categories = get_categories(data[independent_variable])
                if independent_variable not in self._ordinal_variables + self._scale_variables:
                    categories_combo = get_unordered_combinations(categories, 2)
                else:
                    categories_combo = get_ordered_combinations(categories, 2)
                results = pd.DataFrame(columns=categories,
                       index=categories)
                for pair in categories_combo:
                    pair_data = data[data[independent_variable].isin(pair)]
                    if len(pair_data) > 0:
                        anv = ANOVA(pair_data, dependent_variable, independent_variable, show_results=False)
                        var1, var2 = pair
                        results.loc[var1, var2] = anv.pvalue
                        results.loc[var2, var1] = anv.pvalue

                max_pvalue = results[min_count_category].max()
                max_pvalue_category = pd.to_numeric(results[min_count_category]).idxmax()
                if independent_variable not in self._scale_variables:
                    data[independent_variable] = merge_two_cats(data[independent_variable],
                                                                                 min_count_category,
                                                                                 max_pvalue_category)
                else:
                    data[independent_variable] = merge_two_intervals(data[independent_variable],
                                                                                 min_count_category,
                                                                                 max_pvalue_category)          
                min_count = data[independent_variable].value_counts().min()
                n_categories = len(data[independent_variable].unique())
        return data[independent_variable]

    def _build_nodes_from_variable(self,
                                   data,
                                   dependent_variable,
                                   independent_variable=None,
                                   start_counter=0,
                                   parent_node=None):
        
        if independent_variable is not None:
            categories = get_categories(data[independent_variable])
            nodes = []
            counter = start_counter
            n_observations = []
            for category in categories:
                filter_data = data[data[independent_variable]==category][dependent_variable]
                n = len(filter_data)
                n_observations.append(n)
            small_n = [n for n in n_observations if n <= self.min_child_node]    
            if len(small_n) < len(categories) - 1:
                for category in categories:
                    filter_data = data[data[independent_variable]==category][dependent_variable]
                    n = len(filter_data)
                    mean = filter_data.mean()
                    std = filter_data.std()
                    node_info = {'Node': f'Node {counter}',
                                 'Parent node': parent_node,
                                 'Variable': independent_variable,
                                 'Category': category,
                                 'Mean': mean,
                                 'St. dev.': std,
                                 'N': n}
                    nodes.append(node_info)
                    counter += 1
        else:
            filter_data = data[dependent_variable]
            n = len(filter_data)
            mean = filter_data.mean()
            std = filter_data.std()
            node_info = {'Node': 'Node 0',
                         'Variable': dependent_variable,
                         'Mean': mean,
                         'St. dev.': std,
                         'N': n}
            nodes = [node_info]                                
                
        return nodes
    
    def _split_nodes(self,
                     data,
                     dependent_variable,
                     independent_variables,
                     tree_nodes,
                     parent_node='Node 0'):
        
        data_iter = data.copy()
        for variable in independent_variables:
            
            if variable in self._scale_variables:
                data_iter[variable] = binning(data_iter[variable], self.n_intervals)

            data_iter[variable] = self._merging(data_iter,
                                                dependent_variable,
                                                variable)

        split_variable, split_pvalue, split_F, split_dof_b, split_dof_w = CHAIDRegressor\
        ._get_most_significant_variable(data_iter,
                                        dependent_variable,
                                        independent_variables)
        
        if split_variable is not None:
            split_info = {
                         'F': split_F,
                         'p-value': split_pvalue,
                         'dof_b': split_dof_b,
            'dof_w': split_dof_w}

            if split_info['p-value'] <= self.sig_level_split:
                iter_nodes = self._build_nodes_from_variable(data_iter,
                                                                       dependent_variable,
                                                                       split_variable,
                                                                       start_counter=len(tree_nodes),
                                                                       parent_node=parent_node)
                for node in iter_nodes:
                    node.update(split_info)
                                 
                if len(iter_nodes) >= 2:
                    tree_nodes.extend(iter_nodes)
                    current_depths = [CHAIDRegressor._get_node_depth(node, tree_nodes)\
                                        for node in iter_nodes if node['Node']!='Node 0']
                    current_depth = max(current_depths)
                    if current_depth < self.max_depth:
                        for i in range(len(iter_nodes)):
                            parent_node = iter_nodes[i]['Node']
                            node_variable = iter_nodes[i]['Variable']
                            node_category = iter_nodes[i]['Category']
                            node_prediction = iter_nodes[i]['Mean']
                            data_per_node = data[data_iter[node_variable]==node_category].copy()
                            self._data.loc[data_per_node.index, f'{self._dependent_variable} (predicted)'] = node_prediction
                            self._data.loc[data_per_node.index, 'Node'] = parent_node
                            if len(data_per_node) >= self.min_parent_node:
                                available_variables = independent_variables.copy()
                                available_variables.remove(node_variable)
                                tree_nodes = self._split_nodes(data_per_node,
                                                                   dependent_variable,
                                                                   available_variables,
                                                                   tree_nodes,
                                                                   parent_node)
                    else:
                        for i in range(len(iter_nodes)):
                            parent_node = iter_nodes[i]['Node']
                            node_variable = iter_nodes[i]['Variable']
                            node_category = iter_nodes[i]['Category']
                            node_prediction = iter_nodes[i]['Mean']
                            data_per_node = data[data_iter[node_variable]==node_category].copy()
                            self._data.loc[data_per_node.index, f'{self._dependent_variable} (predicted)'] = node_prediction
                            self._data.loc[data_per_node.index, 'Node'] = parent_node
        

        return tree_nodes
                             
    @staticmethod
    def _get_node_depth(node, nodes):
        depth = 0
        while True:
            if 'Parent node' in node:
                parent_node = node['Parent node']
                node = nodes[int(parent_node.split()[-1])]
                depth += 1
            else:
                break
        return depth
    
    @staticmethod
    def _get_one_node_interaction(node, nodes):
        interaction = []
        if node['Node'] == 'Node 0':
            return np.nan
        while True:
            if node['Node'] == 'Node 0':
                break
            interaction.append(f'{node["Variable"]} = {node["Category"]}')
            node = nodes[nodes['Node']==node['Parent node']].iloc[0]
        return ' * '.join(interaction)
                               
    def get_interactions(self, result='dict'):
        """
        Return a dictionary or a DataFrame with nodes and interactions corresponded to them.

        Parameters
        ----------
        result : 'dict' or 'DataFrame'
            Data type of results

        Returns
        -------
        dict or pd.DataFrame
            Definitions of nodes in terms of interactions
        """
                               
        results = self.nodes.apply(lambda x: CHAIDRegressor._get_one_node_interaction(x, self.nodes), axis=1)
        results.index = self.nodes['Node']
        if result.lower() == 'dict':
            return dict(results)
        elif result.lower() == 'dataframe':
            return pd.DataFrame(results, columns=['Interaction'])
        else:
            raise ValueError("Unknown result type. Possible values: 'dict' and 'DataFrame'.")