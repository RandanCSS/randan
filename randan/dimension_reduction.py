from itertools import combinations
from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator
from sklearn.base import BaseEstimator
import scipy as sp
import statsmodels.formula.api as smf
import time


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
                    .format(None, na_rep="", precision=n_decimals)\
                    .set_caption("attribute .inertia_by_dimensions"))
        print('------------------\n')
        print('Detailed information')
        display(self.summary().style\
                    .format(None, na_rep="", precision=n_decimals)\
                    .set_caption("method .summary()"))
    
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

# this class is not for direct use
class Rotator():

    """
    The Rotator class takes an (unrotated) component loading matrix and performs one of several rotations.

    Parameters
    ----------
    method : str, optional
        The component rotation method. Options include:
            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            
            ОСТАЛЬНЫЕ ПОКА НЕ РАБОТАЮТ
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)
        Defaults to 'varimax'.
    normalize : bool or None, optional
        Whether to perform Kaiser normalization and de-normalization prior to and following rotation.
        Used for varimax and promax rotations.
        If None, default for promax is False, and default for varimax is True.
        Defaults to None.
    gamma : int, optional
        The gamma level for the oblimin objective.
        Ignored if the method is not 'oblimin'.
        Defaults to 0.
    kappa : int, optional
        The extention to which raise the components' correlation within 'promax'.
        Numbers generally range form 2 to 4.
        Ignored if the method is not 'equamax'.
        Defaults to 4.
    max_iter : int, optional
        The maximum number of iterations.
        Used for varimax and oblique rotations.
        Defaults to `1000`.
    tol : float, optional
        The convergence threshold.
        Used for varimax and oblique rotations.
        Defaults to `1e-5`.

    Attributes
    ----------
    loadings_ : pandas DataFrame
        The loadings matrix
    rotation_ : numpy array, shape (n_components, n_components)
        The rotation matrix
    psi_ : numpy array or None
        The component correlations
        matrix. This only exists
        if the rotation is oblique.

    Notes
    -----


    References
    ----------
    [1] https://factor-analyzer.readthedocs.io/en/latest/_modules/factor_analyzer/rotator.html

    Examples
    --------
    >>> rotator = RotatorBiggs(ars_list=vars_init_list,PCs_list=PCs_init_list_1,loadings_=loadings_init_df_1)
    >>> rotator.fit_transform()
    DataFrame...
    """

    def __init__(self,vars_list,PCs_list,loadings_,
                 method='varimax',
                 normalize=True,
                 gamma=0,
                 kappa=4,
                 max_iter=500,
                 tol=1e-5):

        self.method = method
        self.normalize = normalize
        self.gamma = gamma
        self.kappa = kappa
        self.max_iter = max_iter
        self.tol = tol
        self.vars_list = vars_list
        self.PCs_list = PCs_list
        self.loadings_ = loadings_
        
        self.rotation_ = None
        self.phi_ = None
        self.X = None

    def _varimax(self, X=None):
        """
        Perform varimax (orthogonal) rotation, with optional Kaiser normalization.

        Parameters
        ----------


        Returns
        -------
        self
        """
        if X is None:
            X = self.loadings_.copy()
        
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix using sqrt of the sum of squares (Kaiser)
#         start = time.time()
        if self.normalize:
            normalized_mtx = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)),1,X.copy())
            X = (X.T/normalized_mtx).T
#         print(f"varimax-normalized_mtx: {time.time()-start}")

        # initialize the rotation matrix to N x N identity matrix
        rotation_mtx = np.eye(n_cols)

#         start = time.time()
        d = 0
        for _ in range(self.max_iter):

            old_d = d

            # take inner product of loading matrix and rotation matrix
            basis = np.dot(X,rotation_mtx)

            # transform data for singular value decomposition
            transformed = np.dot(X.T,basis**3-(1.0/n_rows)*np.dot(basis,np.diag(np.diag(np.dot(basis.T,basis)))))

            # perform SVD on the transformed matrix
            U,S,V = np.linalg.svd(transformed)
            
            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U,V)
            d = np.sum(S)

            # check convergence
            if old_d != 0 and d/old_d < 1+self.tol:
#                 print(f'The rotation converged in {_} iterations')
                break
#             if _== self.max_iter-1:
#                 print(f'The rotation is not converged in {_} iterations')
                
#         print(f"varimax-max_iter: {time.time()-start}")

        # take inner product of loading matrix and rotation matrix
        X = np.dot(X,rotation_mtx)


        # de-normalize the data
#         start = time.time()
        if self.normalize:
            X = X.T*normalized_mtx
        else:
            X = X.T
#         print(f"varimax-DEnormalize: {time.time()-start}")

        # я вписал
        # convert loadings matrix to data frame
        loadings = pd.DataFrame(X.T,columns=self.loadings_.columns,index=self.loadings_.index)
        
        variance = self._get_component_variance(loadings)[0]
        loadings = loadings[variance.sort_values(ascending=False).index]
        
        PCs_vrmx_list = [f'PC{i}_vrmx' for i in range(1, len(self.PCs_list) + 1)]
        loadings.columns = PCs_vrmx_list
        
        return loadings, rotation_mtx

    def _promax(self):
        """
        Perform promax (oblique) rotation, with optional Kaiser normalization.

        Parameters
        ----------


        Returns
        -------
        loadings : numpy array, shape (n_features, n_components)
            The loadings matrix
        rotation_mtx : numpy array, shape (n_components, n_components)
            The rotation matrix
        psi : numpy array or None, shape (n_components, n_components)
            The component correlations
            matrix. This only exists
            if the rotation is oblique.
        """
        X = self.loadings_.copy()
        
        if self.normalize:
            # pre-normalization is done in R's
            # `kaiser()` function when rotate='Promax'.
            array = X.copy()
            h2 = sp.diag(np.dot(array, array.T))
            h2 = np.reshape(h2, (h2.shape[0], 1))
            weights = array / sp.sqrt(h2)

        else:
            weights = X.copy()
        # first get varimax rotation
        X, rotation_mtx = self._varimax(weights)
        Y = X*np.abs(X)**(self.kappa-1)

        # fit linear regression model
        coef = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y))

        # calculate diagonal of inverse square
        try:
            diag_inv = sp.diag(sp.linalg.inv(sp.dot(coef.T,coef)))
        except np.linalg.LinAlgError:
            diag_inv = sp.diag(sp.linalg.pinv(sp.dot(coef.T,coef)))

        # transform and calculate inner products
        coef = sp.dot(coef,sp.diag(sp.sqrt(diag_inv)))
        z = sp.dot(X,coef)

        if self.normalize:
            # post-normalization is done in R's
            # `kaiser()` function when rotate='Promax'
            z = z*sp.sqrt(h2)

        rotation_mtx = sp.dot(rotation_mtx,coef)

        coef_inv = np.linalg.inv(coef)
        phi = np.dot(coef_inv, coef_inv.T)

        # я вписал
        # convert loadings matrix to data frame
        loadings = pd.DataFrame(z, 
                                columns=self.loadings_.columns,
                                index=self.loadings_.index)
        
        variance = self._get_component_variance(loadings)[0]
        loadings = loadings[variance.sort_values(ascending=False).index]
        
        PCs_vrmx_list = [f'PC{i}_prmx' for i in range(1, len(self.PCs_list) + 1)]
        loadings.columns = PCs_vrmx_list
        
        return loadings, rotation_mtx, phi
    
    @staticmethod
    def _get_component_variance(loadings):
        """
        A helper method to get the component variances,
        because sometimes we need them even before the model is fitted.

        Parameters
        ----------
        loadings : array-like
            The component loading matrix,
            in whatever state.

        Returns
        -------
        variance : numpy array
            The component variances.
        proportional_variance : numpy array
            The proportional component variances.
        cumulative_variances : numpy array
            The cumulative component variances.
        """
        n_rows = loadings.shape[0]

        # calculate variance
        loadings = loadings**2
        variance = np.sum(loadings,axis=0)

        # calculate proportional variance
        proportional_variance = variance/n_rows

        # calculate cumulative variance
        cumulative_variance = np.cumsum(proportional_variance, axis=0)

        return (variance,
                proportional_variance,
                cumulative_variance)
        
    def fit(self):
        """
        Computes the component rotation.

        Parameters
        ----------


        Returns
        -------
        self
        """
        self.fit_transform(self.X)
        return self
    
    def fit_transform(self):
        """
        Computes the component rotation,
        and returns the new loading matrix.

        Parameters
        ----------

        Returns
        -------
        loadings_ : pandas DataFrame
            The loadings matrix

        Raises
        ------
        ValueError
            If the `method` is not in the list of
            acceptable methods.

        Example
        -------
        >>> rotator = RotatorBiggs(ars_list=vars_init_list,PC_list=PCs_init_list_1,loadings_=loadings_init_df_1)
        >>> rotator.fit_transform()
        DataFrame...
        """
        # default phi to None
        # it will only be calculated
        # for oblique rotations
        phi = None
        method = self.method.lower()
        if method == 'varimax':
            (new_loadings,
             new_rotation_mtx) = self._varimax()

        elif method == 'promax':
            (new_loadings,
             new_rotation_mtx,
             phi) = self._promax()

#         elif method in OBLIQUE_ROTATIONS:
#             (new_loadings,
#              new_rotation_mtx,
#              phi) = self._oblique(X,method)

#         elif method in ORTHOGONAL_ROTATIONS:
#             (new_loadings,
#              new_rotation_mtx) = self._orthogonal(X,method)

        else:
            raise ValueError("The value for `method` must be one of the "
                             "following: {}.".format(', '.join(POSSIBLE_ROTATIONS)))

        (self.loadings_,
         self.rotation_,
         self.phi_) = new_loadings, new_rotation_mtx, phi
        return self.loadings_

class PCA: # первичен
    
    """A class for implementing principal component analysis (PCA).
    
    Parameters
    ----------
    n_components : int or str 
        The exact number of dimensions
        in solution or the criterion for its automatic selection.
        Current possible values: 'kaiser' (defualt),
        which corresponds to Kaiser's criterion
    rotation : None, 'varimax', 'promax', 'natural collinearity' 
        Rotation to perform on factor loadings.
        Currently, only varimax rotation is available
    kappa : int
        Kappa parameter for promax rotation


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
    
    def __init__(self, n_components='Kaiser', rotation=None, kappa=4):
        if str(n_components).lower() == 'kaiser':
            self.n_components_criterion = 'Kaiser'
        elif str(n_components).lower() == 'inflection':
            self.n_components_criterion = 'Inflection point (based on scree plot)'
        elif isinstance(n_components, int):
            self.n_components = n_components
            self.n_components_criterion = 'User based'
        else:
            raise ValueError(f"""Invalid number of components was passed.
            Possible values: exact number of components, 'Kaiser', or 'Inflection'.""")
            
        possible_rotations = ['varimax', 'natural collinearity', 'promax']
        
        if rotation is not None:
            if rotation.lower() not in possible_rotations:
                raise ValueError(f"Invalid type of rotation was passed. Possible values: {', '.join(possible_rotations)}.")            
            else:
                self.rotation = rotation.lower()
        else:
            self.rotation = None
                                
        self.kappa = kappa
        self._pc_max_list = None
                    
    def fit(self, 
            data, 
            variables=None, 
            scale=True, 
            show_results=True,
            n_decimals=3,
            print_decision=True):
        
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
        print_decision : bool
            Whether to print decision on what number of dimensions was exctracted

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
        init_df = data.copy()
        
        self.variables = list(init_df.columns)
        self.max_n_components = len(self.variables)
        self._PCs_init_list = [f'PC{i+1}' for i in range(self.max_n_components)]
        self.correlation_matrix = pd.DataFrame(np.corrcoef(init_df.T),columns=init_df.columns,index=init_df.columns)
        self.correlation_matrix.index.name = 'Variable'
        self._eigenvectors, self._eigenvalues, self._eigenvectors_t = np.linalg.svd(self.correlation_matrix)
        self._eigenvalues, self._eigenvectors = PCA._sort_sign(self._eigenvalues, self._eigenvectors)
        
        #max_n_components solution
        #start = time.time()
        init_component_loadings = PCA._loadings(self._eigenvalues, 
                                          self._eigenvectors,
                                          self.variables,
                                          self._PCs_init_list)
        #print(f"._loadings(): {time.time()-start}")
        self.explained_variance = self.get_explained_variance(scree_plot=False)
        
        if self.n_components_criterion == 'Kaiser':
            
            self.n_components = len(self.explained_variance[self.explained_variance.iloc[:, 0].astype('float') >= 1])
#             if print_decision:
#                 print(f'The number of selected components by Kaiser criterion: {self.n_components}')
            
        elif self.n_components_criterion == 'Inflection point (based on scree plot)':
            self.n_components = PCA._inflection_point(self.explained_variance)
#             if print_decision:
#                 print(f'The number of selected components by an inflection point: {self.n_components}')
        self.explained_variance_total = self.explained_variance.loc[self.n_components, 'Cumulative %']
        
        #n_components solution
        self.component_loadings = init_component_loadings.iloc[:, :self.n_components]
        self.component_loadings_rotated = None
                             
        self._PCs_final_list = self._PCs_init_list[:self.n_components]
        
        if self.rotation is not None:
            if self.n_components > 1:
                self.component_loadings_rotated,self.structure_matrix = self._rotation_matrices(init_df,
                                                                                 self.n_components,
                                                                                 self.variables,
                                                                                 self._PCs_final_list,
                                                                                 self.component_loadings,
                                                                                 show_results)
    
                self._PCs_final_list = list(self.component_loadings_rotated.columns)
            else:
                print('Rotation could not be performed because number of dimensions is 1.')
                self.rotation = None

        
        if self.rotation in ['natural collinearity', 'promax']:
            necessary_loadings = self.structure_matrix.copy()
        elif self.rotation == 'varimax':
            necessary_loadings = self.component_loadings_rotated.copy()
        else:
            necessary_loadings = self.component_loadings.copy()
            
        #communalities through the component loadings
        self._necessary_loadings = necessary_loadings.copy()
        self.communalities = self.get_communalities(min_max=False)        
        
        self.communalities_and_loadings = pd.concat([self.communalities, necessary_loadings], axis=1)
        
        if self.rotation in ['natural collinearity', 'promax']:
            #start = time.time()     
            self._transformed_data_rotated = self.transform(init_df, add_to_data=False)
            #print(f".transform(): {time.time()-start}")        
            self.correlation_matrix_components = pd.DataFrame(np.corrcoef(self._transformed_data_rotated.T),
                                                          columns=self._transformed_data_rotated.columns,
                                                          index=self._transformed_data_rotated.columns)
            
            pc_corr = self.correlation_matrix_components.copy()
            _pc_max_list = []
            for pc in pc_corr:    
                _pc_max_list.append(abs(pc_corr[pc][(pc_corr[pc])!=1]).max())
            _pc_max_list.sort()
            self._pc_max_list = _pc_max_list.copy()
            
        if show_results:
            self.show_results()
                                 
        return self

    def show_results(self, print_decision=True, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int 
            Number of digits to round results when showing them
        print_decision : bool
            Whether to print decision on what number of dimensions was exctracted
        """
        if self.rotation != 'natural collinearity':
            print('\nPCA SUMMARY')
            print('------------------\n')                                 
            if print_decision:
                if self.n_components_criterion=='Kaiser': 
                    print(f'The number of selected components by Kaiser criterion: {self.n_components}')
                elif self.n_components_criterion=='Inflection point (based on scree plot)':
                    print(f'The number of selected components by an inflection point: {self.n_components}')
                print('------------------\n')
        
        if self.rotation != 'natural collinearity':
            print('Explained variance')
            display(self.get_explained_variance(scree_plot=True).style\
                    .format(None, na_rep="", precision=n_decimals)\
                    .set_caption("methods .get_explained_variance() and .scree_plot()"))
            print(f'The model explains {round(self.explained_variance_total, n_decimals)}% of variance.')
            print('------------------\n')
        if self.rotation is None:
            print('Component loadings')
        elif self.rotation == 'varimax':
            print('Rotated component loadings')
        elif self.rotation in ['natural collinearity', 'promax']:
            print('Structure matrix')
        display(self.communalities_and_loadings.style\
                .format(None, na_rep="", precision=n_decimals)\
                .set_caption("attribute .communalities_and_loadings"))
        print(f'The minimum communality is {round(self.communalities["Communality"].min(), n_decimals)}.')
        if self.rotation in ['natural collinearity', 'promax']:
            print("Components' correlation")
            display(self.correlation_matrix_components.style\
                .format(None, na_rep="", precision=n_decimals)\
                .set_caption("attribute .correlation_matrix_components"))
        print('------------------\n')
        print('To get component scores, use [model].transform().')
    def transform(self, data, standardize=True, add_to_data=False):
        
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
        
        df = data.copy()
        df = (df - df.mean()) / df.std()
        if self.rotation is not None and self.rotation == 'natural collinearity':
            return PCA._nat_collinearity(data, self.n_components, show=False, add_to_data=add_to_data, loadings_matrix=True)[3]
        else:                     
            loadings_df = self._necessary_loadings.copy()
                             
        df.columns = loadings_df.index
        try:
            weights = np.linalg.solve(self.correlation_matrix, loadings_df)
            #display(weights)
        except:
            weights = loadings_df.copy()
        component_scores_df = pd.DataFrame(np.dot(df, weights), 
                                           index = df.index,
                                           columns = loadings_df.columns)
        if standardize:
            component_scores_df = (component_scores_df - component_scores_df.mean()) / component_scores_df.std()
            
        if add_to_data:
            component_scores_df = pd.concat([df, component_scores_df], axis=1)

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
#         print('S',S,'np.sqrt(S)',np.sqrt(S))
        loadings_df = pd.DataFrame(loadings)
        loadings_df.columns = PCs_list
        loadings_df.index = vars_list
        loadings_df.index.name = 'Variable'
        return loadings_df
        
    def _rotation_matrices(self, data, n_components, vars_list, PCs_list, loadings_, show):
        if self.rotation == 'natural collinearity':
            matrices = PCA._nat_collinearity(data, n_components, show=show, add_to_data=False, loadings_matrix=True)
            component_loadings_rotated = matrices[1]
            structure_matrix = matrices[2]
            #print("rotation == 'natural collinearity'")

        else:
#             start = time.time()
            rotator = Rotator(vars_list, PCs_list, loadings_, kappa=self.kappa)
#             print(f"Rotator(): {time.time()-start}")        
            
            if self.rotation == 'varimax':
                loadings,rotation_mtx = rotator._varimax()            
#                 signs = np.sign(loadings.sum(0))
#                 signs[(signs==0)] = 1
#                 loadings = np.dot(loadings,np.diag(signs))
#                 component_loadings_rotated = pd.DataFrame(loadings,
#                                                  columns=loadings_.columns,
#                                                  index=loadings_.index)
                component_loadings_rotated = loadings
                structure_matrix = None
                #print("rotation == 'varimax'")
            
            else:
#                 start = time.time()
                loadings,rotation_mtx,phi = rotator._promax()
#                 print(f"_promax(): {time.time()-start}")
#                 signs = np.sign(loadings.sum(0))
#                 signs[(signs==0)] = 1
#                 loadings = np.dot(loadings,np.diag(signs))
#                 phi = np.dot(np.dot(np.diag(signs),phi),np.diag(signs))
                component_loadings_rotated = loadings
#                 start = time.time()
                structure_matrix = np.dot(loadings,phi)
#                 print(f".dot(): {time.time()-start}")
#                 start = time.time()
                structure_matrix = pd.DataFrame(structure_matrix,
                                                 columns=loadings.columns,
                                                 index=loadings.index)
#                 print(f".DataFrame(): {time.time()-start}")
                #print("rotation == 'promax'")

        return component_loadings_rotated,structure_matrix
    
    @staticmethod
    def _inflection_point(explained_variance, tol=0.05):
        percent_df = pd.DataFrame(columns=['Num','Percent'],
                                  index=range(1, len(explained_variance)+1))
        percent_df['Num'] = range(1, len(explained_variance)+1)
        percent_df['Percent'] = (explained_variance.iloc[:, 1]).astype('float')
        inflection_df = pd.DataFrame(columns=['delta'])
        for i in range(len(percent_df) - 1,0,-1): # running from the right point on the graph to the left
            results = smf.ols('Percent~Num', data=percent_df.loc[i:len(percent_df), :]).fit()
            a = results.rsquared
            results = smf.ols('Percent~Num', data=percent_df.loc[i - 1:len(percent_df), :]).fit()
            b = results.rsquared
            inflection_df.loc[i, 'delta'] = a - b

        inflection_point = inflection_df[inflection_df['delta'] == inflection_df['delta'].max()].index[0] - 1
        #print(f'The number of selected components by an inflection point = {inflection_point}')
        return inflection_point
#         delta = 0
#         i = len(percent_df) - 1
#         while delta < tol:
#             results = smf.ols('Percent ~ Num', data=percent_df.loc[i:len(percent_df), :]).fit()
#             a = results.rsquared
#             results = smf.ols('Percent ~ Num', data=percent_df.loc[i-1:len(percent_df), :]).fit()
#             b = results.rsquared
#             delta = a - b
#             i -= 1
        
#         return i
                             
    @staticmethod
    def _nat_collinearity(data, n_components, show=True, add_to_data=False, loadings_matrix=False):
        vrmx_model = PCA(n_components=n_components, rotation='varimax').fit(data, print_decision=False, show_results=False)                         
        dist_2 = vrmx_model.variables_components_distribution(show=False)[1]                         
        component_loadings = pd.DataFrame(index=data.columns)
        component_scores = pd.DataFrame(index=data.index)
        eigenvalue = 0
        i = 0
        for i_v in dist_2['Variables']:
            i += 1
            model = PCA().fit(data[i_v], print_decision=False, show_results=False)
            component_loadings[f'PC{i}_natcol'] = model.component_loadings['PC1']# component loadings -- МОГУТ ВЫВОДИТЬСЯ ПО СПЕЦ.ЗАПРОСУ
            eigenvalue += model.get_explained_variance(scree_plot=show if i < 5 else False)['Eigenvalue'][1]
            component_scores[f'PC{i}_natcol'] = model.transform(data[i_v])
        pc_corr = pd.DataFrame(np.corrcoef(component_scores.T),
                               columns=component_scores.columns,
                               index=component_scores.columns) # correlation of the components
        component_scores = pd.concat([data, component_scores],axis=1)
        structure_matrix = component_scores.corr()
        structure_matrix = structure_matrix.loc[data.columns, pc_corr.columns]

        if add_to_data==False:
            component_scores = component_scores.drop(data.columns, axis=1)

        if show:
            print(f'The total eigenvalue explained by Natural collinearity PCA (i.e., {len(dist_2)} independent PCA-models) = {round(eigenvalue,3)}',
            f'The total variance explained by Natural collinearity PCA = {round(100*eigenvalue/len(data.columns),3)}%',
            '', sep='\n')   

        if loadings_matrix:
            return pc_corr, component_loadings, structure_matrix, component_scores
        else:
            return pc_corr, structure_matrix, component_scores

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
        vars_list = self.variables.copy()
        PCs_list = self._PCs_final_list.copy()
        communalities = pd.DataFrame(index=loadings.index)
        communalities['Communality'] = (loadings**2).sum(axis=1)
        if min_max:
            print(f"The min communality: {round(communalities['Communality'].min(), 3)}, the max communality: {round(communalities['Communality'].max(), 3)}")
        return communalities
    
    def get_explained_variance(self, n_decimals=3, scree_plot=True):
        
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
                  
        S = self._eigenvalues.copy()
        PCs_list = self._PCs_init_list.copy()
        exp = 100 * S / np.sum(S)
        acc_sum = np.cumsum(exp)
        explained_variance = np.array([S, exp, acc_sum])
        explained_variance = pd.DataFrame(explained_variance.T, 
                                   columns=['Eigenvalue','Variance accounted for, %','Cumulative %'],
                                   index = range(1, len(PCs_list)+1))
        explained_variance.index.name = 'Component'
        explained_variance = round(explained_variance, n_decimals)

        if scree_plot:
            self.scree_plot()
        
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
        
    def variables_components_distribution(self,
                                          threshold=0.5,
                                          delta=0.1,
                                          show=True):
                  
        """
        Return several useful tables (as tuple) that may help to interpret solution:
        the distribution of variables through components,
        the distribution of components through variables,
        the competing components within variables.  
        
        Parameters
        ----------
        threshold : float 
            What value of an absolute component loading 
            to use when deciding about which variable belongs mostly to which component
            (default: 0.5)
        
        delta : float 
            What value of an absolute component loading 
            to consider as big enough to exclude component from competing
            (default: 0.1)
        
        show : bool 
            Whether to display all dataframes at once in a convinient way
            (default: True)

        Returns
        -------
        tuple
            A sequence of the requested dataframes
        """
        if self.n_components < 2:
            print('The distributions would not be shown because number of dimensions is 1.')
        else:
            loadings_df = self._necessary_loadings.copy()
            PCs_list = self._PCs_final_list.copy()

            PCs_through_vars = loadings_df.copy() # the components distribution through variables (loading >= 0.5 by module)
            for PC in PCs_list:
                PCs_through_vars[PC] = PCs_through_vars[PC].apply(lambda loading: '+' if abs(loading)>=threshold else '')

            vars_through_PCs = loadings_df.copy() # the variables distribution through components (loading >= 0.5 by module)
            vars_through_PCs_list = []
            for PC in PCs_list:
                vars_candidates_plus = list(vars_through_PCs[PC][vars_through_PCs[PC]>=threshold].index)
                vars_candidates_minus = list(vars_through_PCs[PC][vars_through_PCs[PC]<=-threshold].index)
                #if vars_candidates != []:
                vars_through_PCs_list.append([PC, vars_candidates_plus, vars_candidates_minus])
            vars_through_PCs = pd.DataFrame(vars_through_PCs_list)
            vars_through_PCs.columns = ['Component', 'Variables (positive side)', 'Variables (negative side)']
            vars_through_PCs.set_index('Component', inplace=True)

            #переименовать переменные для лучшей читаемости
            try:
                loadings_df_3 = loadings_df.copy() # the competing components within variables (loading >= 0.5 by module & difference < 0.1)
                for PC in PCs_list:
                    loadings_df_3[PC] = loadings_df_3[PC].apply(lambda loading: abs(loading) if abs(loading)>=threshold else None)
                loadings_df_3_1 = loadings_df_3.T
                v_PC_multi_list_3 = []
                for var in loadings_df_3_1.columns:
                    loadings_df_3_2 = loadings_df_3_1[var].copy()
                    loadings_df_3_2 = loadings_df_3_2.sort_values(ascending=False)
                    n_competing_components = loadings_df_3_2.count()
                    if n_competing_components > 1:
                        if loadings_df_3_2[0]-loadings_df_3_2[1] >= delta:
                            v_PC_multi_list_3.append([var, 
                                                      n_competing_components, 
                                                      (loadings_df_3_2[loadings_df_3_2 == loadings_df_3_2[0]].index)[0],
                                                      None])
                        else:
                        # БЕРУ ТОЛЬКО 2 КОНКУРИРУЮЩИЕ КОМПОНЕНТЫ (т.к. при вращении вряд ли потребуется больше)
                            try:
                                v_PC_multi_list_3.append([var, 
                                                          n_competing_components, 
                                                          (loadings_df_3_2[loadings_df_3_2 == loadings_df_3_2[0]].index)[0],
                                                          (loadings_df_3_2[loadings_df_3_2 == loadings_df_3_2[1]].index)[0]])
                            except:
                                v_PC_multi_list_3.append([var, 
                                                          0, 
                                                          None,
                                                          None])
                  
                    else:
                        try:
                            v_PC_multi_list_3.append([var, 
                                                  n_competing_components, 
                                                  (loadings_df_3_2[loadings_df_3_2 == loadings_df_3_2[0]].index)[0],
                                                  None])
                        except:
                            v_PC_multi_list_3.append([var, 
                                                          0, 
                                                          None,
                                                          None])
                v_PC_multi_df_3 = pd.DataFrame(v_PC_multi_list_3,
                                               columns=['Variable',
                                                        'N of competing components',
                                                        'Stronger component',
                                                        'Weaker component'])
                v_PC_multi_df_3.set_index('Variable', inplace=True)
            except:
                v_PC_multi_df_3 = None

            if show:
                print('The fist dataframe represents the distribution of variables across components.')
                display(PCs_through_vars)
                print('-------------')
                print('The second dataframe represents the distribution of components across variables.')      
                display(vars_through_PCs)
                print('-------------')
                print(f'''The third dataframe represents the competing components within variables 
                (which loadings >= {threshold} & difference between loadings <= {delta})''')
                display(v_PC_multi_df_3)

            return PCs_through_vars, vars_through_PCs, v_PC_multi_df_3


# previous version
# class PCA:
    
#     """
#     A class for implementing principal component analysis (PCA).
    
#     Parameters
#     ----------
#     n_components : int or str 
#         The exact number of dimensions
#         in solution or the criterion for its automatic selection.
#         Current possible values: 'kaiser' (defualt),
#         which corresponds to Kaiser's criterion
#     rotation : None or 'varimax' 
#         Rotation to perform on factor loadings.
#         Currently, only varimax rotation is available

#     Attributes
#     ----------
#     correlation_matrix : pd.DataFrame
#         A correlation matrix
#     explained_variance : pd.DataFrame
#         A table with eigenvalues and variance accounted for
#     explained_variance_total : float
#         Total percentage of the explained variance
#     component_loadings : pd.DataFrame
#         Component (factor) loadings
#     component_loadings_rotated : pd.DataFrame
#         Component (factor) loadings after rotation
#     communalities : pd.DataFrame
#         Communalities
#     communalities_and_loadings : pd.DataFrame
#         A joint table of component (factor) loadings and communalities
#     """
    
#     def __init__(self, n_components='Kaiser', rotation=None):
#         if str(n_components).lower() == 'kaiser':
#             self.n_components_criterion = 'Kaiser'
#         elif isinstance(n_components, int):
#             self.n_components = n_components
#             self.n_components_criterion = 'User based'
#         else:
#             raise ValueError(f"""Invalid number of components was passed.
#             Possible values: exact number of components or 'Kaiser'.""")
            
#         possible_rotations = ['varimax']
        
#         if rotation is not None:
#             if rotation.lower() not in possible_rotations:
#                 phrase = ', '.join(possible_rotations)
#                 raise ValueError(f"Invalid type of rotation was passed. Possible values: {phrase}.")

#             else:
#                 self.rotation = rotation.lower()
#         else:
#             self.rotation = None
                                
#         #self.kappa = kappa
#         #self._pc_max_list = None
                    
#     def fit(
#         self, 
#         data, 
#         variables=None,
#         scale=True,
#         show_results=True, 
#         n_decimals=3
#     ):
        
#         """
#         Fit a model to the given data.
        
#         Parameters
#         ----------
        
#         data : pd.DataFrame 
#             Data to fit a model
#             variables (None or list): variables from data to include in a model.
#             If not specified, all variables will be used.
#         variables : list
#             Names of variables from data that should be used in a model.
#             If not specified, all variables from data are used. Variables should have a numeric dtype.
#         scale : bool 
#             Whether data should be considered as scale variables. 
#             If set to False, data will be transformed to ranks. 
#         show_results :bool 
#             Whether to show results of the analysis
#         n_decimals : int 
#             Number of digits to round results when showing them

#         Returns
#         -------
#         self
#             The current instance of the PCA class
#         """    
        
#         if variables is not None:
#             data = data[variables].dropna()
#         else:
#             data = data.dropna()
                                 
#         if not scale:
#             data = data.rank()
                                 
#         self._data = data.copy()
#         self.variables = list(data.columns)
#         self.max_n_components = len(self.variables)
#         self._PCs_init_list = [f'PC{i+1}' for i in range(self.max_n_components)]
#         self.correlation_matrix = pd.DataFrame(
#             np.corrcoef(data.T),
#             columns=data.columns,
#             index=data.columns
#         )
#         self.correlation_matrix.index.name = 'Variable'
                                 
#         self._eigenvectors, self._eigenvalues, self._eigenvectors_t = np.linalg.svd(self.correlation_matrix)
#         self._eigenvalues, self._eigenvectors = PCA._sort_sign(self._eigenvalues, self._eigenvectors)
        
#         #max_n_components solution
#         init_component_loadings = PCA._loadings(self._eigenvalues, 
#                                           self._eigenvectors,
#                                           self.variables,
#                                           self._PCs_init_list)
                                 
#         self.explained_variance = self.get_explained_variance(scree_plot=False)
        
#         if self.n_components_criterion == 'Kaiser':
            
#             self.n_components = len(self.explained_variance[self.explained_variance.iloc[:, 0].astype('float') >= 1])
                                 
#         self.explained_variance_total = self.explained_variance.loc[self.n_components, 'Cumulative %']
        
#         #n_components solution
#         self.component_loadings = init_component_loadings.iloc[:, :self.n_components]
#         self.component_loadings_rotated = None
                             
#         self._PCs_final_list = self._PCs_init_list[:self.n_components]
        
#         if self.rotation is not None:
#             if self.n_components > 1:
#                 self.component_loadings_rotated, self.structure_matrix = self._rotation_matrices()
    
#                 self._PCs_final_list = list(self.component_loadings_rotated.columns)
#             else:
#                 print('Rotation could not be performed because number of dimensions is 1.')
#                 self.rotation = None

#         if self.rotation == 'varimax':
#             necessary_loadings = self.component_loadings_rotated.copy()
#         else:
#             necessary_loadings = self.component_loadings.copy()
            
#         #communalities through the component loadings
#         self._necessary_loadings = necessary_loadings.copy()
#         self.communalities = self.get_communalities(min_max=False)        
        
#         self.communalities_and_loadings = pd.concat([necessary_loadings, self.communalities], axis=1)
            
#         if show_results:
#             self.show_results(n_decimals=n_decimals)
                                 
#         return self

#     def show_results(self, n_decimals=3):
#         """
#         Show results of the analysis in a readable form.
        
#         Parameters
#         ----------
#         n_decimals : int 
#             Number of digits to round results when showing them
#         """
#         print('\nPCA SUMMARY')
#         print('------------------\n')                                 
#         if self.n_components_criterion=='Kaiser':
#             print(f'The number of selected components by Kaiser criterion: {self.n_components}')
#             print('------------------\n')
#         print('Explained variance')
#         display(self.get_explained_variance(scree_plot=True).style\
#                     .format(None, na_rep="", precision=n_decimals)\
#                     .set_caption("methods .get_explained_variance() and .scree_plot()"))
#         print(f'The model explains {round(self.explained_variance_total, 3)}% of variance.')
#         print('------------------\n')
#         if self.rotation is None:
#             print('Component loadings')
#         elif self.rotation == 'varimax':
#             print('Rotated component loadings')
#         display(self.communalities_and_loadings.style\
#                 .format(None, na_rep="", precision=n_decimals)\
#                 .set_caption("attribute .communalities_and_loadings"))
#         print(f'The minimum communality is {round(self.communalities["Communality"].min(), 3)}.')
#         print('------------------\n')
#         print('To get component scores, use [model].transform().')

#     def transform(self, data=None, standardize=True, add_to_data=False):
        
#         """
#         Return component scores for every observation in the given dataset. 
        
#         Parameters
#         ----------
#         data : pd.DataFrame 
#             Data to apply the model.
#             If not specified, data that were used to fit the model will be used.
#         standardize : bool 
#             Whether to apply z-standartization to component scores
#         add_to_data : bool 
#             Whether to add variables of component scores to the given data

#         Returns
#         -------
#         pd.DataFrame
#             Requested values
#         """
#         if data is None:
#             data = self._data.copy()
#             df = self._data.copy()
#         else:
#             df = data[self.variables].dropna().copy()
              
#         df = (df - df.mean()) / df.std()
#         loadings_df = self._necessary_loadings.copy()                     
#         df.columns = loadings_df.index
                  
#         try:
#             weights = np.linalg.solve(self.correlation_matrix, loadings_df)
#         except:
#             weights = loadings_df.copy()
#         component_scores_df = pd.DataFrame(np.dot(df, weights), 
#                                            index = df.index,
#                                            columns = loadings_df.columns)
#         if standardize:
#             component_scores_df = (component_scores_df - component_scores_df.mean()) / component_scores_df.std()
            
#         if add_to_data:
#             component_scores_df = pd.concat([data, component_scores_df], axis=1)

#         return component_scores_df
        
#     @staticmethod
#     def _sort_sign(S, U):
#         idx = S.argsort()[::-1]
#         S = S[idx]
#         U = U[:, idx]
#         for i in range(U.shape[1]):
#             if U[:, i].sum()<0:
#                 U[:, i] = -1 * U[:, i]
#         return S, U
    
#     @staticmethod
#     def _loadings(S, U, vars_list, PCs_list):
#         loadings = U * np.sqrt(S)
#         loadings_df = pd.DataFrame(loadings)
#         loadings_df.columns = PCs_list
#         loadings_df.index = vars_list
#         loadings_df.index.name = 'Variable'
#         return loadings_df
        
#     def _rotation_matrices(self):
#         if self.rotation == 'varimax':
#             rot = Rotator(method='varimax', normalize=True)
#             loadings = rot.fit_transform(self.component_loadings)
#             loadings = pd.DataFrame(loadings,
#                                    columns=[f'PC{i+1}_vrmx' for i in range(self.n_components)],
#                                    index=self.variables)
#             structure_matrix = None

#         return loadings, structure_matrix


#     def get_communalities(self, min_max=True):
        
#         """
#         Return communalities for every initial variable. 
        
#         Parameters
#         ----------
#         min_max : bool 
#             Whether to print minimum and maximum of communalities

#         Returns
#         -------
#         pd.DataFrame
#             A table with communalities
#         """
#         loadings = self.component_loadings.copy()
#         communalities = pd.DataFrame(index=loadings.index)
#         communalities['Communality'] = (loadings**2).sum(axis=1)
         
#         if min_max:
#             min_ = round(communalities['Communality'].min(), 3)
#             max_ = round(communalities['Communality'].max(), 3)
#             print(f"The min communality: {min_}, the max communality: {max_}")
        
#         return communalities
    
#     def get_explained_variance(self, scree_plot=True, **kwargs):
        
#         """
#         Return summary table with information about variance accounted for. 
        
#         Parameters
#         ----------
#         scree_plot : bool 
#             Whether to display scree plot        
#         annotate_bars : bool 
#             Whether to annotate exact percentage of variance on each bar
#             (if scree_plot set to True)
#         annotate_current : bool 
#             Whether to show percentage of variance corresponded 
#             to the current solution
#             (if scree_plot set to True)

#         Returns
#         -------
#         pd.DataFrame
#             A table with explained variance
#         """          
                  
#         S = self._eigenvalues
#         PCs_list = self._PCs_init_list
#         exp = 100 * S / np.sum(S)
#         acc_sum = np.cumsum(exp)
#         explained_variance = np.array([S, exp, acc_sum])
#         explained_variance = pd.DataFrame(explained_variance.T, 
#                                    columns=['Eigenvalue','Variance accounted for, %','Cumulative %'],
#                                    index = range(1, len(PCs_list)+1))
#         explained_variance.index.name = 'Component'

#         if scree_plot:
#             self.scree_plot(**kwargs)
        
#         return explained_variance
    
#     def scree_plot(self, annotate_bars=True, annotate_current=True):
                  
#         """
#         Vizualize distribution of the variance accounted for.  
        
#         Parameters
#         ----------
#         annotate_bars : bool 
#             Whether to annotate exact percentage of variance on each bar
#         annotate_current : bool 
#             Whether to show percentage of variance corresponded 
#             to the current solution
#         """ 
            
#         explained_variance = self.explained_variance.copy()    
#         plt.figure(figsize=(6, 6))
        
#         pc_num = list(explained_variance.index)
#         each_exp = explained_variance.iloc[:, 1].tolist()
#         acc_sum = explained_variance.iloc[:, 2].tolist()
#         plt.bar(pc_num, acc_sum, width=0.5, color='lightsalmon', alpha=0.2, label='Cumulative %')
#         plt.plot(pc_num, each_exp, label='Variance accounted for, %')
#         plt.plot(pc_num, each_exp, 'ro', label='_nolegend_')
        
#         if annotate_bars:
#             for x, y in zip(pc_num, acc_sum):
#                 plt.annotate(f'{round(y)}%', (x-0.25, y+0.75))
        
#         if annotate_current:
#             acc_sum_by_n_components = [explained_variance.iloc[self.n_components-1, 2]] * len(pc_num)
#             plt.plot(pc_num, acc_sum_by_n_components, linestyle='--', label='Current solution', c='black')
                
#         plt.xlabel('Principal components')
#         plt.ylabel('Variance accounted for, %')
#         plt.title('Variance accounted by components', fontsize=16)
#         plt.legend(loc='upper left')
#         plt.show()
