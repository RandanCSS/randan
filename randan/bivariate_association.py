import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, norm
from IPython.display import display
from scipy.stats import pearsonr, spearmanr, kendalltau
from pandas.api.types import is_numeric_dtype


class Crosstab:
    """
    
    Class to perform analysis of contingency tables based on Pearson's chi-square.
    
    Parameters
    ----------
    data : pd.DataFrame 
        Data to build a contingency table on
    row : str 
        Exact variable from the data to create rows of a contingency table
    column : str 
        Exact variable from the data to create columns of a contingency table
    sig_level : float
        Level of significance to analyze Pearson's residuals
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int
        Number of digits to round results when showing them
    only_stats : bool 
        Use this if you only want to get final statistics,
        i.e. chi2, p-value, dof, and expected frequencies

    Attributes
    ----------
    N : int
        Number of observations used in the analysis
    n_cells : int
        Number of cells in a contingency table 
    frequencies_observed : pd.DataFrame
        Observed frequencies
    frequencies_expected : pd.DataFrame
        Expected frequencies
    residuals : pd.DataFrame
        'Raw' residuals 
        (differences between obs. and exp. frequencies)
    residuals_pearson : pd.DataFrame
        Pearson's residuals
        (raw residuals divided by square root of expected frequencies)
    chi_square: float
        Chi-square statistics
    dof : int
        Degrees of freedom
    pvalue : float
        P-value of a chi-square statistic
    row_categories : list
        Names of categories located at rows of a contingency table
    column_categories : list
        Names of categories located at columns of a contingency table
    row_marginals : pd.DataFrame
        Marginal frequencies of row categories
    column_marginals : pd.DataFrame
        Marginal frequencies of column categories
    """        
    
    def __init__(self, 
                 data, 
                 row=None, 
                 column=None,
                 sig_level=0.05,
                 show_results=True,
                 n_decimals=3,
                 only_stats=False):
              
        if row is not None and column is not None:
            self.data = data[[row, column]]
        elif (row is None or column is None) and len(data.columns) > 2:
            raise ValueError('''Number of variables in dataframe exceeds 2.
            Please, specify both row and column arguments or filter the necessary variables manually.''')
        elif len(data.columns) < 2:
            raise ValueError('''One or no variables were passed.''')
        else:
            self.data = data
        
        self.sig_level = sig_level
        
        self._frequencies_observed = pd.crosstab(self.data.iloc[:, 0],
                                                self.data.iloc[:, 1])
        
        if only_stats:
            self.chi_square, self.pvalue, self.dof, self.frequencies_expected = \
            chi2_contingency(self._frequencies_observed)
            return
        
        self.N = self._frequencies_observed.sum().sum()
        
        self.n_cells = self._frequencies_observed.shape[0] * self._frequencies_observed.shape[1]        
        
        self.row_categories = list(self._frequencies_observed.index)
        self.column_categories = list(self._frequencies_observed.columns)
        
        self.frequencies_observed = self._frequencies_observed.copy()
        
        self.row_marginals = Crosstab._get_marginals(self._frequencies_observed, 'row')       
        self.column_marginals = Crosstab._get_marginals(self._frequencies_observed, 'column')
        
        self.frequencies_observed = Crosstab._add_marginals(self._frequencies_observed,
                                                            self.row_marginals,
                                                            self.column_marginals,
                                                            self.N)

        self._frequencies_expected = self._get_expected_frequencies()
               
        self.frequencies_expected = Crosstab._add_marginals(self._frequencies_expected,
                                                            self.row_marginals,
                                                            self.column_marginals,
                                                            self.N)
        self.residuals = self._frequencies_observed - self._frequencies_expected
        self.residuals_pearson = self.residuals / (self._frequencies_expected**0.5)
        self.chi_square = (self.residuals_pearson**2).sum().sum()
        self.dof = (len(self.row_categories) - 1)*(len(self.column_categories) - 1)
               

        self.pvalue = chi2.sf(self.chi_square, self.dof)
        
        if show_results:
            self.show_results(n_decimals)
    
    @staticmethod
    def _get_marginals(crosstab, axis='row'):
        if axis=='row':            
            marginals = pd.DataFrame(crosstab.sum(axis=1),
                                     columns=['Total'])        
        elif axis=='column':
            marginals = pd.DataFrame(crosstab.sum(),
                                     columns=['Total']).T
        else:
            raise ValueError('Unknown axis.')
        
        return marginals    
    
    @staticmethod
    def _add_marginals(crosstab, row_marginals, column_marginals, N=None):
        if N is None:
            N = crosstab.sum().sum()
        
        crosstab = crosstab.copy()
        
        #weird but the only way that works though        
        try:
            crosstab.columns = crosstab.columns.add_categories('Total')
            crosstab['Total'] = row_marginals
        except AttributeError:
            crosstab['Total'] = row_marginals
        
        try:    
            crosstab.index = crosstab.index.add_categories('Total')
            crosstab.loc['Total'] = column_marginals.T['Total'].tolist() + [N]
        except AttributeError:
            crosstab.loc['Total'] = column_marginals.T['Total'].tolist() + [N]        
        
        return crosstab
    
    def _get_expected_frequencies(self):
        frequencies_expected = np.array([self.column_marginals.T['Total'].to_list()] * len(self.row_categories))
        frequencies_expected = frequencies_expected * self.row_marginals['Total'].to_numpy()[:, np.newaxis] / self.N
        frequencies_expected = pd.DataFrame(frequencies_expected,
                                            columns=self._frequencies_observed.columns,
                                            index=self._frequencies_observed.index)
        return frequencies_expected
        
    def check_small_counts(self, n_decimals=3):
        """
        Check sparsity of the crosstab, i.e., 
        calculate percentage of small expected frequencies
        (those that are less than 5).

        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them

        """
        
        n_small_counts = (self._frequencies_expected < 5).sum().sum()
        min_count = self._frequencies_expected.min().min()
        n_cells = self.n_cells
        perc_small_counts = n_small_counts / n_cells * 100
        print(f'''{n_small_counts} ({round(perc_small_counts, n_decimals)}%) cells have expected \
frequency less than 5. The minimum expected frequency is {round(min_count, n_decimals)}.''')
        
    def check_significant_residuals(self, sig_level=0.05, n_decimals=3):
        """
        Identify significant Pearson's residuals 
        based on the given significant level.

        Parameters
        ----------
        sig_level : float
            Significance level (alpha) to identify significant residuals
        n_decimals : int
            Number of digits to round results when showing them
        """
        
        critical_value = norm.isf(sig_level / 2)
        n_sig_resids = (abs(self.residuals_pearson) >= critical_value).sum().sum()
        n_cells = self.n_cells
        perc_sig_resids = n_sig_resids / n_cells * 100
        
        max_resid = self.residuals_pearson.max().max()
        max_resid_row = self.residuals_pearson.max(axis=1).idxmax()
        max_resid_column = self.residuals_pearson.max(axis=0).idxmax()
        
        min_resid = self.residuals_pearson.min().min()
        min_resid_row = self.residuals_pearson.min(axis=1).idxmin()
        min_resid_column = self.residuals_pearson.min(axis=0).idxmin()
        print(f'''{n_sig_resids} ({round(perc_sig_resids, n_decimals)}%) cells have Pearson's \
residual bigger than {round(critical_value, 2)}. 
The biggest residual is {round(max_resid, n_decimals)} (categories {max_resid_row} and {max_resid_column}).
The smallest residual is {round(min_resid, n_decimals)} (categories {min_resid_row} and {min_resid_column}).''')
        
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.

        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        
        sig_level = self.sig_level
        
        print('\nCROSSTAB SUMMARY')
        print('------------------\n')
        print('Observed frequencies')
        display(self.frequencies_observed.style\
                    .format(None, na_rep="")\
                    .set_caption("attribute .frequencies_observed")\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('Expected frequencies')
        display(self.frequencies_expected.style\
                    .format(None, na_rep="")\
                    .set_caption("attribute .frequencies_expected")\
                    .set_precision(n_decimals))        
        self.check_small_counts(n_decimals)
        print('------------------\n')
        print(f'Chi-square statistic is {round(self.chi_square, n_decimals)} (p-value = {round(self.pvalue, n_decimals)}).')
        print('------------------\n')
        print("Pearson's residuals")
        display(self.residuals_pearson.style\
                    .format(None, na_rep="")\
                    .set_caption("attribute .residuals_pearson")\
                    .set_precision(n_decimals))
        self.check_significant_residuals(sig_level, n_decimals)

#todo: flag significant, pairwise deletion

class Correlation:
    """
    
    Class to perform correlation analysis.
    
    Parameters
    ----------
    data : pd.DataFrame 
        Data to perform the analysis
    variables : list
        Variables from data to include in the analysis
    method : str
        Which correlation coefficient to use.
        Available values: 'pearson', 'spearman', 'kendall'
    two_tailed : bool
        Whether to calculate two-tailed p-value (set to False if you are interested in one-tailed p-value)
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int
        Number of digits to round results when showing them
    min_max : bool
        Whether to print minimum and maximum correlation coefficients
    Attributes
    ----------
    tailes : int
        Number of tailes (1 or 2) to estimate p-value
    method : int
        Type of correlation coefficient  
    N : int
        Number of observations included in the analysis
    correlation_matrix : pd.DataFrame
        Correlation matrix
    """ 
    def __init__(
        self,
        data,
        variables=None,
        method='pearson',
        two_tailed=True,
        show_results=True,
        n_decimals=3, 
        min_max=True
    ):
        if variables is not None and not isinstance(variables, list):
            raise TypeError(f'Variables should be passed as list. Type {type(variables)} was passed instead.')
            
        if not isinstance(method, str):
            raise TypeError(f'Method should be passed as str. Type {type(method)} was passed instead.')
            
        if variables is None:
            variables = data.columns
        
        non_num_vars = [var for var in variables if not is_numeric_dtype(data[var])]
        zero_var_vars = [var for var in variables if is_numeric_dtype(data[var]) and data[var].var()==0]
        variables = [var for var in variables if var not in non_num_vars and var not in zero_var_vars]
        
        self._non_num_vars = []
        self._zero_var_vars = []
        
        if len(non_num_vars) > 0:
            self._non_num_vars = non_num_vars
        if len(zero_var_vars) > 0:
            self._zero_var_vars = zero_var_vars    
        
        if len(variables) < 2:
            raise ValueError('One or zero numeric variables were passed or found in a dataframe.')
        
        data = data[variables].dropna().copy()
        
        self.tailes = 2 if two_tailed else 1
        
        if method.lower() == 'pearson':
            metric = pearsonr
        elif method.lower() == 'spearman':
            metric = spearmanr
        elif method.lower() == 'kendall':
            metric = kendalltau
#         elif method.lower() == 'all':
#             pass
        else:
            raise ValueError("Unknown method requested. Possible values: 'pearson', 'spearman', 'kendall'")
            
        self.method = method
        
        N = len(data)
        self.N = N
        
        stats = ['Coefficient', 'p-value', 'N']
        index = [variables, stats]
        index = pd.MultiIndex.from_product(index)
        results = pd.DataFrame(index=index)
        
        corr_dict = {}
        
        for var_1 in variables:
            for var_2 in variables:
                if var_1 == var_2:
                    r, pval = 1, np.nan
                else:
                    r, pval = metric(data[var_1], data[var_2])
                    if not two_tailed:
                        pval /= 2
                    if (var_2, var_1) not in corr_dict.keys():
                        corr_dict.update({(var_1, var_2): (r, pval)})
                
                results.loc[(var_1, 'Coefficient'), var_2] = r
                results.loc[(var_1, 'p-value'), var_2] = pval
                results.loc[(var_1, 'N'), var_2] = N
        
        self._corr_dict = corr_dict
        
        self.correlation_matrix = results
        
        if show_results:
            self.show_results(n_decimals, min_max)
            
    def show_results(self, n_decimals=3, min_max=True):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        min_max : bool
            Whether to print minimum and maximum correlation coefficients
        """
        print(f'\nCORRELATION SUMMARY ({self.method.upper()} METHOD, {self.tailes}-TAILED)')
        print('------------------')
        
        if len(self._non_num_vars) > 0:
            phrase = 'The following variables were removed from the analysis since they do not belong to numerical dtypes: {}\n'
            print(phrase.format(', '.join(self._non_num_vars)))
        if len(self._zero_var_vars) > 0:
            phrase = 'The following variables were removed from the analysis since they have zero variance: {}\n'
            print(phrase.format(', '.join(self._zero_var_vars)))
        display(self.correlation_matrix.style\
                    .format(None, na_rep="")\
                    .set_caption("attribute .correlation_matrix")\
                    .set_precision(n_decimals))
        if min_max:
            results = self.sort_correlations()
            self.min_corr = results['Coefficient'].min()
            idxmin = results['Coefficient'].idxmin()
            min_pval = round(results.loc[idxmin, 'p-value'], n_decimals)
            self.max_corr = results['Coefficient'].max()
            idxmax = results['Coefficient'].idxmax()
            max_pval = round(results.loc[idxmax, 'p-value'], n_decimals)
            print(f'Maximum correlation is {round(self.max_corr, n_decimals)} (p-value {max_pval}) for variables {idxmax[0]} and {idxmax[1]},',
                  f'minimum correlation is {round(self.min_corr, n_decimals)} (p-value {min_pval}) for variables {idxmin[0]} and {idxmin[1]}.', 
                  sep='\n')
            
    def sort_correlations(self, by='coefficient', ascending=True):
        """
        Sort correlations between all possible pairs of variables.

        Parameters
        ----------
        by : str
            Which parameter should define sorting: 'coefficient' or 'pvalue'
        ascending : bool
            Whether to sort values in ascending order
        """
        coefs = pd.DataFrame(self._corr_dict, index=[0, 1]).T
        coefs.columns = ['Coefficient', 'p-value']
        if by.lower() == 'coefficient':
            by = 'Coefficient'
        elif by.lower() in ['p-value', 'pvalue', 'p_value']:
            by = 'p-value'
        print('Note: Each empty index duplicates the previous one.')
        return coefs.sort_values(by=by, ascending=ascending)
