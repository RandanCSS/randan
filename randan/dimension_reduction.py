from itertools import combinations
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator

from IPython.display import display

class CA:
    
    """
    A class for implementing correspondence analysis (CA),
    a method for identifying associations of two categorical
    variables.
    
    Parameters
    ----------
    n_dimensions : int 
        A number of dimensions in solution (equals 2 by default)
    
    Attributes
    ----------
    crosstab : pd.DataFrame
        A contingency table
    correspondence_matrix : pd.DataFrame
        A normalized (divided by number of observations) contingency table 
    row_categories : list
        Names of the categories located at the rows of a crosstab
    column_categories : list
        Names of the categories located at the columns of a crosstab
    row_mass : pd.Series
        Masses of row categories
    column_mass : pd.Series
        Masses of column categories
    inertia_total : float
        Total inertia of a crosstab
    inertia_by_dimensions : pd.DataFrame
        Inertia accounted for
    inertia_of_rows : np.ndarray
        Inertia of row categories
    inertia_of_columns : np.ndarray
        Inertia of column categories
    principal_row_coordinates : np.ndarray
        Principal coordinates of row categories
    principal_column_coordinates : np.ndarray
        Principal coordinates of column categories
    standard_row_coordinates : np.ndarray
        Standard coordinates of row categories
    standard_column_coordinates : np.ndarray
        Standard coordinates of column categories
    contribution_of_rows_to_inertia_of_dimensions : np.ndarray
        Contribution of row categories to inertia of dimensions
    contribution_of_columns_to_inertia_of_dimensions : np.ndarray
        Contribution of column categories to inertia of dimensions
    contribution_of_dimensions_to_inertia_of_rows : np.ndarray
        Contribution of dimensions to inertia of row categories 
    contribution_of_dimensions_to_inertia_of_columns : np.ndarray
        Contribution of dimensions to inertia of column categories
    communalities_rows : np.ndarray
        Analogues of communalities for row categories
    communalities_columns : np.ndarray
        Analogues of communalities for column categories
    component_loadings_rows : pd.DataFrame
        Analogues of component (factor) loadings for row categories
    component_loadings_columns : pd.DataFrame
        Analogues of component (factor) loadings for column categories
    """
    
    def __init__(self, n_dimensions=2):
        self.n_dimensions = n_dimensions    

    def fit(
        self, 
        data, 
        row=None, 
        column=None, 
        data_type='raw',
        show_results=True,
        n_decimals=3,
        plot_dimensions=True
    ):
        
        """
        Fit a model to the given data.
        
        Parameters
        ----------
        data : pd.DataFrame 
            Data to fit a model. 
            Either the raw data or the contingency table with row categories as index
        row : str 
            [only necessary if raw data are passed]
            Name of a variable of the data to be considered as row values
        column : str 
            [only necessary if raw data are passed]
            Name of a variable of the data to be considered as column values
        data_type : str 
            Type of the data passed, 
            possible values are 'raw' or 'crosstab' 
        show_results : bool 
            Whether to show results of the analysis
        n_decimals : int 
            Number of digits to round results when showing them
        plot_dimensions : bool 
            Whether to vizualize two first dimensions

        Returns
        -------
        self
            The current instance of the CA class
        """
        
        if data_type.lower() == 'raw':
            if len(data.columns) > 2 and (not row and not column):
                raise ValueError("""Number of variables in DataFrame
                exceeds 2. Please specify both row and column arguments
                or filter the necessary variables manually.""")

            elif len(data.columns) < 2:
                raise ValueError("""One or no variables were passed.
                Please add necessary variables to DataFrame.""")

            if row and column:
                self.crosstab = pd.crosstab(data[row], data[column])

            elif row or column:
                raise ValueError("""Please specify both row and column arguments
                or filter the necessary variables manually.""")

            else: 
                self.crosstab = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])

        elif data_type.lower() == 'crosstab':
            
            self.crosstab = data.copy()
            
        else:
            raise ValueError("""Unknown data type. Possible values: 'raw' or 'crosstab'""")
            
        while len(self.crosstab.loc[self.crosstab.var(axis=1) == 0.0, :]) > 0 or \
        len(self.crosstab.loc[:, self.crosstab.var() == 0.0].T) > 0:
            self.crosstab = self.crosstab.loc[:, self.crosstab.var() > 0.0].copy()
            self.crosstab = self.crosstab.loc[self.crosstab.var(axis=1) > 0.0, :].copy()
            print('One or more categories were removed from analysis as they have zero variance.')
        
        max_inertia = min(len(self.crosstab.columns),
                         len(self.crosstab.index)) - 1
        
        if self.n_dimensions > max_inertia:
            raise ValueError(f"""Too many dimensions requested. The maximum possible number of
            dimensions is {max_inertia}, but {self.n_dimensions} were requested.""")
        
        self.N = self.crosstab.sum().sum()
        self.column_marginals = self.crosstab.sum()
        self.row_marginals = self.crosstab.sum(axis=1)
        self.correspondence_matrix = self.crosstab / self.N
        self.row_categories = list(self.correspondence_matrix.index)
        self.column_categories = list(self.correspondence_matrix.columns)
        
        
        self.column_profiles = self.crosstab / self.column_marginals
        
        try:
            self.column_profiles.columns = self.column_profiles.columns.add_categories('Mean column profile')
            self.column_profiles['Mean column profile'] = self.row_marginals / self.N
        except AttributeError:
            self.column_profiles['Mean column profile'] = self.row_marginals / self.N

        
        self.row_profiles = (self.crosstab.T / self.row_marginals).T
        
        try:
            self.row_profiles.index = self.row_profiles.index.add_categories('Mean row profile')
            self.row_profiles.loc['Mean row profile'] = self.column_marginals / self.N
        except AttributeError:
            self.row_profiles.loc['Mean row profile'] = self.column_marginals / self.N
            

        
        self.row_mass = self.column_profiles['Mean column profile'].copy()
        self.row_mass.name = 'Row mass'
        self.column_mass = self.row_profiles.loc['Mean row profile'].copy()
        self.column_mass.name = 'Column mass'
        
        self.expected_frequencies = pd.DataFrame(np.outer(self.correspondence_matrix.sum(),
                                                          self.correspondence_matrix.sum(axis=1)).T,
                                                 index = self.correspondence_matrix.index,
                                                 columns = self.correspondence_matrix.columns)
        
        self.standardized_residuals = (self.correspondence_matrix                                        - self.expected_frequencies) / np.sqrt(self.expected_frequencies)
        
        self.inertia_total = (self.standardized_residuals ** 2).sum().sum()
        
        self.U, self.singular_values, self.V = np.linalg.svd(self.standardized_residuals, full_matrices=True)
        
        self.inertia_by_dimensions = pd.DataFrame(self.singular_values ** 2,
                                                        columns = ['Inertia'],
                                                        index = [f'Dimension {i+1}' for i in range(len(self.singular_values))])
        self.inertia_by_dimensions['Inertia (%)'] = self.inertia_by_dimensions['Inertia'] / self.inertia_total * 100
        self.inertia_by_dimensions = self.inertia_by_dimensions.iloc[:self.n_dimensions, :]
        self.inertia_by_dimensions.loc['Total'] = self.inertia_by_dimensions.sum()
        
        self.standard_row_coordinates = (np.diag(self.row_mass.values**-0.5) @ self.U)[:, :max_inertia]
        self.standard_column_coordinates = np.diag(self.column_mass.values**-0.5) @ self.V[:max_inertia, :].T
        
        self.principal_row_coordinates = self.standard_row_coordinates @ np.diag(self.singular_values[:max_inertia])
        self.principal_column_coordinates = np.diag(self.column_mass.values**-0.5) @ self.V[:max_inertia, :].T @ np.diag(self.singular_values[:max_inertia])
        
        self.row_inertia_components = np.diag(self.row_mass.values) @ self.principal_row_coordinates**2
        self.column_inertia_components = np.diag(self.column_mass.values) @ self.principal_column_coordinates**2
        
        self.contribution_of_rows_to_inertia_of_dimensions = self.row_inertia_components / self.row_inertia_components.sum(axis=0)[np.newaxis, :]
        self.contribution_of_columns_to_inertia_of_dimensions = self.column_inertia_components / self.column_inertia_components.sum(axis=0)[np.newaxis, :]
        
        self.inertia_of_rows = self.row_inertia_components.sum(axis=1)
        self.inertia_of_columns = self.column_inertia_components.sum(axis=1)
        
        self.contribution_of_dimensions_to_inertia_of_rows = self.row_inertia_components / self.inertia_of_rows[:, np.newaxis]
        self.contribution_of_dimensions_to_inertia_of_columns = self.column_inertia_components / self.inertia_of_columns[:, np.newaxis]
        
        self.communalities_rows = self.contribution_of_dimensions_to_inertia_of_rows[:, :self.n_dimensions].sum(axis=1)
        self.communalities_columns = self.contribution_of_dimensions_to_inertia_of_columns[:, :self.n_dimensions].sum(axis=1)
        
        self.component_loadings_rows = self.contribution_of_dimensions_to_inertia_of_rows**0.5 * np.sign(self.principal_row_coordinates)
        self.component_loadings_rows = pd.DataFrame(self.component_loadings_rows[:, :self.n_dimensions], 
                                                columns = [f'Dimension {i+1}' for i in range(self.n_dimensions)],
                                                index = self.correspondence_matrix.index)
        self.component_loadings_columns = self.contribution_of_dimensions_to_inertia_of_columns**0.5 * np.sign(self.principal_column_coordinates)
        self.component_loadings_columns = pd.DataFrame(self.component_loadings_columns[:, :self.n_dimensions], 
                                                columns = [f'Dimension {i+1}' for i in range(self.n_dimensions)],
                                                index = self.correspondence_matrix.columns)
        
        if show_results:
            self.show_results(n_decimals)
            
        if plot_dimensions:
            print('------------------\n')
            if self.n_dimensions == 1:
                print("The plot couldn't be drawn with n_demnsions set to 1.")
            else:
                print("Dimension's plot")
                self.plot_dimensions()
                if self.n_dimensions > 2:
                    print('Only two first dimensions are shown.')
                    print(f'To get more plots, use [model].plot_dimensions((1, {self.n_dimensions}))')
        
        return self
    
    def show_results(self, n_decimals):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int 
            Number of digits to round results when showing them
        """
        print('\nCA SUMMARY')
        print('------------------\n')
        print('Explained inertia')
        display(self.inertia_by_dimensions.style\
                    .format(None, na_rep="")\
                    .set_caption("attribute .inertia_by_dimensions")\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('Detailed information')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption("method .summary()")\
                    .set_precision(n_decimals))        
    
    def summary(self, 
                display_component_loadings=True, 
                display_contributions=False,
                display_coordinates=False):
        """
        Return summary of the fitted model
        in terms of mass, inertia, contributions and communalities.
        
        Parameters
        ----------
        display_component_loadings : bool 
            Whether to display component loadings 
            instead of 'raw' contributions of dimensions and points to inertia in the final table
        display_contributions : bool 
            Whether to display 
            'raw' contributions of dimensions and points in the final table
        display_coordinates : bool 
            Whether to display 
            coordinates of points in the final table

        Returns
        -------
        pd.DataFrame
            A summary table
        """
        
        summary_table = pd.DataFrame(index = self.row_categories + self.column_categories)
        summary_table['Mass'] = list(self.row_mass) + list(self.column_mass)
        
        if display_coordinates==True:
            for i in range(self.n_dimensions):
                summary_table[f'Coordinates (dim. {i+1})'] = list(self.principal_row_coordinates[:, i]) + list(self.principal_column_coordinates[:, i])
        
        summary_table['Inertia (%)'] = list(self.inertia_of_rows / self.inertia_total * 100) + list(self.inertia_of_columns / self.inertia_total * 100)
        
        if display_contributions==True:
            for i in range(self.n_dimensions):
                summary_table[f'Contribution of points to inertia of dimensions (dim. {i+1})'] = list(self.contribution_of_rows_to_inertia_of_dimensions[:, i]) + list(self.contribution_of_columns_to_inertia_of_dimensions[:, i])
            
            for i in range(self.n_dimensions):
                summary_table[f'Contribution of dimensions to inertia of points (dim. {i+1})'] = list(self.contribution_of_dimensions_to_inertia_of_rows[:, i]) + list(self.contribution_of_dimensions_to_inertia_of_columns[:, i])
            
        if display_component_loadings==True:
            for i in range(self.n_dimensions):
                summary_table[f'Component loadings of points (dim. {i+1})'] = list(self.component_loadings_rows.iloc[:, i]) + list(self.component_loadings_columns.iloc[:, i])        
         
        summary_table[f'Communalities ({self.n_dimensions} dims)'] = list(self.communalities_rows) + list(self.communalities_columns)
        
        summary_table.fillna(0, inplace=True)
        
        return summary_table
    
    def show_components(self, axis=0, n_indicators=None):
        
        """
        Show component loadings in a more concise way.
        Useful when dealing with big crosstabs 
        (e.g., when CA is used to text data for topic modeling).
        
        Parameters
        ----------
        axis : int or str 
            Categories of which axis to show, 
            possible values are 'row' (0) or 'column' (1)
        n_indicators : int 
            How many indicators to show, if not specified,
            all indicators will be shown

        Returns
        -------
        pd.DataFrame
            A table with indicators and their loadings
        """
        row_categories = [str(category) + '__ROW' for category in self.row_categories]
        column_categories = [str(category) + '__COLUMN' for category in self.column_categories]
        
        if axis == 0 or axis == 'row':
            categories_to_show = row_categories.copy()
            
        elif axis == 1 or axis == 'column':
            categories_to_show = column_categories.copy()
            
        else:
            raise ValueError("Unknown axis. Possible values: 'row' (0) and 'column' (1)")
        
        if not n_indicators:
            n_indicators = len(categories_to_show)
            
        components = pd.DataFrame(index = range(n_indicators))
        n_components = self.n_dimensions
        summary = self.summary().copy()
        summary.index = row_categories + column_categories
        
        for i in range(2, n_components+2):
            positive_loadings_index = list(summary.loc[categories_to_show].iloc[:, i].sort_values(ascending=False)[:n_indicators].index)
            positive_loadings_values = list(summary.loc[categories_to_show].iloc[:, i].sort_values(ascending=False)[:n_indicators])
            positive_loadings = list(zip(positive_loadings_index, positive_loadings_values))
            positive_loadings = [str(loading)+'*'+index for index, loading in positive_loadings]    
            components[f'Component {i-1} (positive loadings)'] = positive_loadings

            negative_loadings_index = list(summary.loc[categories_to_show].iloc[:, i].sort_values(ascending=True)[:n_indicators].index)
            negative_loadings_values = list(summary.loc[categories_to_show].iloc[:, i].sort_values(ascending=True)[:n_indicators])
            negative_loadings = list(zip(negative_loadings_index, negative_loadings_values))
            negative_loadings = [str(loading)+'*'+index for index, loading in negative_loadings]
            components[f'Component {i-1} (negative loadings)'] = negative_loadings
            
        for component in components.columns:
            components[component] = components[component].str.replace('__ROW', '').str.replace('__COLUMN', '')
        
        return components
    
    def plot_dimensions(
        self, 
        dimensions_range=(1, 2), 
        size=100, 
        n_symbols=20, 
        save=False
    ):

        """
        Build plots of categories's coordinates in the obtained dimensions.
        
        Parameters
        ----------
        dimensions_range : tuple 
            Which dimensions to consider,
            e.g. if dimensions_range is (1, 3), it will produce three plots
            (the 1st dimension & the 2nd dimension,
            the 2nd dimension & the 3rd dimension,
            the 1st dimension & the 3rd dimension)
        size : int 
            Size of plots, in percentage (100 corresponds to figsize=(8, 5))
        n_symbols : int 
            How many symbols of labels to display
        save : bool 
            Whether to save plots or not (in the current directory)
        """
        
        if self.n_dimensions == 1:
            raise ValueError("The plot couldn't be drawn with n_demnsions set to 1.")
        
        dimension_low = dimensions_range[0]-1
        dimension_high = dimensions_range[1]-1

        dimension_combinations = list(combinations(range(dimension_low, dimension_high+1), 2))

        n_plots = len(dimension_combinations)
        size = size/100
        for plot in range(n_plots):
            plt.figure(figsize=(8*size, 5*size))
            current_combination_x, current_combination_y = dimension_combinations[plot]
            x1, y1 = self.principal_row_coordinates[:, current_combination_x], self.principal_row_coordinates[:, current_combination_y]
            x2, y2 = self.principal_column_coordinates[:, current_combination_x], self.principal_column_coordinates[:, current_combination_y]
            plt.scatter(x1, y1, c='r')
            plt.scatter(x2, y2, c='gray', marker='s')

            x_delta = 0.005
            y_delta = 0.005

            for i, txt in enumerate(self.row_categories):
                plt.annotate(txt[:n_symbols]+'...', (x1[i]-x_delta, y1[i]+y_delta))

            for i, txt in enumerate(self.column_categories):
                plt.annotate(txt[:n_symbols]+'...', (x2[i]-x_delta, y2[i]+y_delta))

            plt.axhline(0, alpha=0.2, linestyle='--')  #horizontal line
            plt.axvline(0, alpha=0.2, linestyle='--')
            x_min = min(plt.xticks()[0])
            x_max = max(plt.xticks()[0])

            y_min = min(plt.yticks()[0])
            y_max = max(plt.yticks()[0])

            xticks = plt.xticks()[0]

            plt.fill_between(xticks, y_min, 0,
                            where = xticks <= 0,
                            alpha=0.2)
            plt.fill_between(xticks, y_min, 0,
                            where = xticks >= 0,
                            alpha=0.2)
            plt.fill_between(xticks, 0, y_max,
                            where = xticks <= 0,
                            alpha=0.2)
            plt.fill_between(xticks, 0, y_max,
                            where = xticks >= 0,
                            alpha=0.2);
            plt.xlabel(f'Dimension {current_combination_x+1}')
            plt.ylabel(f'Dimension {current_combination_y+1}')
            plt.grid(alpha=0.15)
            if save:
                plt.savefig(f'Plot_{plot}.png')
            plt.show();

class PCA:
    
    """
    A class for implementing principal component analysis (PCA).
    
    Parameters
    ----------
    n_components : int or str 
        The exact number of dimensions
        in solution or the criterion for its automatic selection.
        Current possible values: 'kaiser' (defualt),
        which corresponds to Kaiser's criterion
    rotation : None or 'varimax' 
        Rotation to perform on factor loadings.
        Currently, only varimax rotation is available

    Attributes
    ----------
    correlation_matrix : pd.DataFrame
        A correlation matrix
    explained_variance : pd.DataFrame
        A table with eigenvalues and variance accounted for
    explained_variance_total : float
        Total percentage of the explained variance
    component_loadings : pd.DataFrame
        Component (factor) loadings
    component_loadings_rotated : pd.DataFrame
        Component (factor) loadings after rotation
    communalities : pd.DataFrame
        Communalities
    communalities_and_loadings : pd.DataFrame
        A joint table of component (factor) loadings and communalities
    """
    
    def __init__(self, n_components='Kaiser', rotation=None):
        if str(n_components).lower() == 'kaiser':
            self.n_components_criterion = 'Kaiser'
        elif isinstance(n_components, int):
            self.n_components = n_components
            self.n_components_criterion = 'User based'
        else:
            raise ValueError(f"""Invalid number of components was passed.
            Possible values: exact number of components or 'Kaiser'.""")
            
        possible_rotations = ['varimax']
        
        if rotation is not None:
            if rotation.lower() not in possible_rotations:
                phrase = ', '.join(possible_rotations)
                raise ValueError(f"Invalid type of rotation was passed. Possible values: {phrase}.")

            else:
                self.rotation = rotation.lower()
        else:
            self.rotation = None
                                
        #self.kappa = kappa
        #self._pc_max_list = None
                    
    def fit(
        self, 
        data, 
        variables=None,
        scale=True,
        show_results=True, 
        n_decimals=3
    ):
        
        """
        Fit a model to the given data.
        
        Parameters
        ----------
        
        data : pd.DataFrame 
            Data to fit a model
            variables (None or list): variables from data to include in a model.
            If not specified, all variables will be used.
        variables : list
            Names of variables from data that should be used in a model.
            If not specified, all variables from data are used. Variables should have a numeric dtype.
        scale : bool 
            Whether data should be considered as scale variables. 
            If set to False, data will be transformed to ranks. 
        show_results :bool 
            Whether to show results of the analysis
        n_decimals : int 
            Number of digits to round results when showing them

        Returns
        -------
        self
            The current instance of the PCA class
        """    
        
        if variables is not None:
            data = data[variables].dropna()
        else:
            data = data.dropna()
                                 
        if not scale:
            data = data.rank()
                                 
        self._data = data.copy()
        self.variables = list(data.columns)
        self.max_n_components = len(self.variables)
        self._PCs_init_list = [f'PC{i+1}' for i in range(self.max_n_components)]
        self.correlation_matrix = pd.DataFrame(
            np.corrcoef(data.T),
            columns=data.columns,
            index=data.columns
        )
        self.correlation_matrix.index.name = 'Variable'
                                 
        self._eigenvectors, self._eigenvalues, self._eigenvectors_t = np.linalg.svd(self.correlation_matrix)
        self._eigenvalues, self._eigenvectors = PCA._sort_sign(self._eigenvalues, self._eigenvectors)
        
        #max_n_components solution
        init_component_loadings = PCA._loadings(self._eigenvalues, 
                                          self._eigenvectors,
                                          self.variables,
                                          self._PCs_init_list)
                                 
        self.explained_variance = self.get_explained_variance(scree_plot=False)
        
        if self.n_components_criterion == 'Kaiser':
            
            self.n_components = len(self.explained_variance[self.explained_variance.iloc[:, 0].astype('float') >= 1])
                                 
        self.explained_variance_total = self.explained_variance.loc[self.n_components, 'Cumulative %']
        
        #n_components solution
        self.component_loadings = init_component_loadings.iloc[:, :self.n_components]
        self.component_loadings_rotated = None
                             
        self._PCs_final_list = self._PCs_init_list[:self.n_components]
        
        if self.rotation is not None:
            if self.n_components > 1:
                self.component_loadings_rotated, self.structure_matrix = self._rotation_matrices()
    
                self._PCs_final_list = list(self.component_loadings_rotated.columns)
            else:
                print('Rotation could not be performed because number of dimensions is 1.')
                self.rotation = None

        if self.rotation == 'varimax':
            necessary_loadings = self.component_loadings_rotated.copy()
        else:
            necessary_loadings = self.component_loadings.copy()
            
        #communalities through the component loadings
        self._necessary_loadings = necessary_loadings.copy()
        self.communalities = self.get_communalities(min_max=False)        
        
        self.communalities_and_loadings = pd.concat([necessary_loadings, self.communalities], axis=1)
            
        if show_results:
            self.show_results(n_decimals=n_decimals)
                                 
        return self

    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int 
            Number of digits to round results when showing them
        """
        print('\nPCA SUMMARY')
        print('------------------\n')                                 
        if self.n_components_criterion=='Kaiser':
            print(f'The number of selected components by Kaiser criterion: {self.n_components}')
            print('------------------\n')
        print('Explained variance')
        display(self.get_explained_variance(scree_plot=True).style\
                    .format(None, na_rep="")\
                    .set_caption("methods .get_explained_variance() and .scree_plot()")\
                    .set_precision(n_decimals))
        print(f'The model explains {round(self.explained_variance_total, 3)}% of variance.')
        print('------------------\n')
        if self.rotation is None:
            print('Component loadings')
        elif self.rotation == 'varimax':
            print('Rotated component loadings')
        display(self.communalities_and_loadings.style\
                .format(None, na_rep="")\
                .set_caption("attribute .communalities_and_loadings")\
                .set_precision(n_decimals))
        print(f'The minimum communality is {round(self.communalities["Communality"].min(), 3)}.')
        print('------------------\n')
        print('To get component scores, use [model].transform().')

    def transform(self, data=None, standardize=True, add_to_data=False):
        
        """
        Return component scores for every observation in the given dataset. 
        
        Parameters
        ----------
        data : pd.DataFrame 
            Data to apply the model.
            If not specified, data that were used to fit the model will be used.
        standardize : bool 
            Whether to apply z-standartization to component scores
        add_to_data : bool 
            Whether to add variables of component scores to the given data

        Returns
        -------
        pd.DataFrame
            Requested values
        """
        if data is None:
            data = self._data.copy()
            df = self._data.copy()
        else:
            df = data[self.variables].dropna().copy()
              
        df = (df - df.mean()) / df.std()
        loadings_df = self._necessary_loadings.copy()                     
        df.columns = loadings_df.index
                  
        try:
            weights = np.linalg.solve(self.correlation_matrix, loadings_df)
        except:
            weights = loadings_df.copy()
        component_scores_df = pd.DataFrame(np.dot(df, weights), 
                                           index = df.index,
                                           columns = loadings_df.columns)
        if standardize:
            component_scores_df = (component_scores_df - component_scores_df.mean()) / component_scores_df.std()
            
        if add_to_data:
            component_scores_df = pd.concat([data, component_scores_df], axis=1)

        return component_scores_df
        
    @staticmethod
    def _sort_sign(S, U):
        idx = S.argsort()[::-1]
        S = S[idx]
        U = U[:, idx]
        for i in range(U.shape[1]):
            if U[:, i].sum()<0:
                U[:, i] = -1 * U[:, i]
        return S, U
    
    @staticmethod
    def _loadings(S, U, vars_list, PCs_list):
        loadings = U * np.sqrt(S)
        loadings_df = pd.DataFrame(loadings)
        loadings_df.columns = PCs_list
        loadings_df.index = vars_list
        loadings_df.index.name = 'Variable'
        return loadings_df
        
    def _rotation_matrices(self):
        if self.rotation == 'varimax':
            rot = Rotator(method='varimax', normalize=True)
            loadings = rot.fit_transform(self.component_loadings)
            loadings = pd.DataFrame(loadings,
                                   columns=[f'PC{i+1}_vrmx' for i in range(self.n_components)],
                                   index=self.variables)
            structure_matrix = None

        return loadings, structure_matrix


    def get_communalities(self, min_max=True):
        
        """
        Return communalities for every initial variable. 
        
        Parameters
        ----------
        min_max : bool 
            Whether to print minimum and maximum of communalities

        Returns
        -------
        pd.DataFrame
            A table with communalities
        """
        loadings = self.component_loadings.copy()
        communalities = pd.DataFrame(index=loadings.index)
        communalities['Communality'] = (loadings**2).sum(axis=1)
         
        if min_max:
            min_ = round(communalities['Communality'].min(), 3)
            max_ = round(communalities['Communality'].max(), 3)
            print(f"The min communality: {min_}, the max communality: {max_}")
        
        return communalities
    
    def get_explained_variance(self, scree_plot=True, **kwargs):
        
        """
        Return summary table with information about variance accounted for. 
        
        Parameters
        ----------
        scree_plot : bool 
            Whether to display scree plot        
        annotate_bars : bool 
            Whether to annotate exact percentage of variance on each bar
            (if scree_plot set to True)
        annotate_current : bool 
            Whether to show percentage of variance corresponded 
            to the current solution
            (if scree_plot set to True)

        Returns
        -------
        pd.DataFrame
            A table with explained variance
        """          
                  
        S = self._eigenvalues
        PCs_list = self._PCs_init_list
        exp = 100 * S / np.sum(S)
        acc_sum = np.cumsum(exp)
        explained_variance = np.array([S, exp, acc_sum])
        explained_variance = pd.DataFrame(explained_variance.T, 
                                   columns=['Eigenvalue','Variance accounted for, %','Cumulative %'],
                                   index = range(1, len(PCs_list)+1))
        explained_variance.index.name = 'Component'

        if scree_plot:
            self.scree_plot(**kwargs)
        
        return explained_variance
    
    def scree_plot(self, annotate_bars=True, annotate_current=True):
                  
        """
        Vizualize distribution of the variance accounted for.  
        
        Parameters
        ----------
        annotate_bars : bool 
            Whether to annotate exact percentage of variance on each bar
        annotate_current : bool 
            Whether to show percentage of variance corresponded 
            to the current solution
        """ 
            
        explained_variance = self.explained_variance.copy()    
        plt.figure(figsize=(6, 6))
        
        pc_num = list(explained_variance.index)
        each_exp = explained_variance.iloc[:, 1].tolist()
        acc_sum = explained_variance.iloc[:, 2].tolist()
        plt.bar(pc_num, acc_sum, width=0.5, color='lightsalmon', alpha=0.2, label='Cumulative %')
        plt.plot(pc_num, each_exp, label='Variance accounted for, %')
        plt.plot(pc_num, each_exp, 'ro', label='_nolegend_')
        
        if annotate_bars:
            for x, y in zip(pc_num, acc_sum):
                plt.annotate(f'{round(y)}%', (x-0.25, y+0.75))
        
        if annotate_current:
            acc_sum_by_n_components = [explained_variance.iloc[self.n_components-1, 2]] * len(pc_num)
            plt.plot(pc_num, acc_sum_by_n_components, linestyle='--', label='Current solution', c='black')
                
        plt.xlabel('Principal components')
        plt.ylabel('Variance accounted for, %')
        plt.title('Variance accounted by components', fontsize=16)
        plt.legend(loc='upper left')
        plt.show()