import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, norm
from IPython.display import display

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