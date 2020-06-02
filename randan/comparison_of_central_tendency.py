import pandas as pd
import numpy as np
from scipy.stats import f
from .utils import get_categories
from IPython.display import display

class ANOVA:
    """
    Class to perform analysis of variance (AnOVa).

    Parameters
    ----------
    data : pd.DataFrame
        Data used to perform the analysis
    dependent_variables : str or list 
        Name(s) of (a) scale dependent variable(s).
        Note that if several variables are used, only summary table is available as a result.
    independent_variable : str 
        Name of an independent (factor, grouping) variable
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int 
        Number of digits to round results when showing them

    Attributes
    ----------
    SSb : float
        Sum of squares between groups
    SSw : float
        Sum of squares within groups
    SSt : float
        Total sum of squares
    MSb : float
        Mean squares between groups
    MSw : float
        Mean squares within groups
    F : float
        F-statistic
    pvalue : float
        P-value of the F-statistic
    dof_b : int
        Degrees of freedom between groups
    dof_w : int
        Degrees of freedom within groups
    dof_t : int
        Total degrees of freedom
    """
    def __init__(self, 
                 data, 
                 dependent_variables, 
                 independent_variable,
                 show_results=True,
                 n_decimals=3):
        
        self._data = data.copy()
        self._dependent_variables = dependent_variables
        self._independent_variable = independent_variable
        
        if isinstance(dependent_variables, list) and len(dependent_variables) > 1:
            self._several_variables_used = True
            self._summary_several_variables = self._perform_anova_for_several_variables()
        
        else:
            if isinstance(dependent_variables, list) and len(dependent_variables) == 1:
                dependent_variables = dependent_variables[0]
                
            self._several_variables_used = False
            data = data[[dependent_variables, independent_variable]].dropna()
            #replace with get_categories from utils
            groups = get_categories(data[independent_variable])
            #print(groups)
            groups_n = len(groups)
            n = len(data)

            dof_w = n - groups_n
            dof_b = groups_n - 1
            dof_t = dof_w + dof_b

            data.set_index(independent_variable, inplace=True)
            groups_means = data.groupby(independent_variable).agg(['mean', 'count'])
            data['mean'] = groups_means[dependent_variables]['mean']

            SSw = ((data[dependent_variables] - data['mean'])**2).sum()

            if dof_w > 0:
                MSw = SSw / dof_w
            else:
                MSw = np.nan

            groups_means['grand_mean'] = data[dependent_variables].mean()

            SSb = (((groups_means[dependent_variables]['mean'] - groups_means['grand_mean'])**2)*groups_means[dependent_variables]['count']).sum()
            MSb = SSb / dof_b

            SSt = SSw + SSb
            if pd.isnull(MSw):
                F = 0
                pvalue = 1
            else:
                F = MSb / MSw
                pvalue = f.sf(F, dof_b, dof_w)
            
            if pd.isnull(pvalue):
                pvalue = 1

            self.SSb = SSb
            self.SSw = SSw
            self.SSt = SSt
            self.MSb = MSb
            self.MSw = MSw
            self.F = F
            self.pvalue = pvalue
            self.dof_b = dof_b
            self.dof_w = dof_w
            self.dof_t = dof_t
        
        if show_results:
            self.show_results(n_decimals=n_decimals)
        
    def summary(self):
        """
        Get summary information on the conducted analysis.

        Returns
        -------
        pd.DataFrame
            Summary table with results of analysis
        """
        if self._several_variables_used:
            
            results = self._summary_several_variables
        
        else:
        
            results = [[self.SSb, self.dof_b, self.MSb, self.F, self.pvalue],
                      [self.SSw, self.dof_w, self.MSw, np.nan, np.nan],
                      [self.SSt, self.dof_t, np.nan, np.nan, np.nan]]
            results = pd.DataFrame(results,
                                  columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'p-value'],
                                  index = ['Between Groups', 'Within Groups', 'Total'])
        
        return results
    
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """

        
        print('\nANOVA SUMMARY')
        print('------------------')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption("method .summary()")\
                    .set_precision(n_decimals))
        
    def _perform_anova_for_several_variables(self):
        summary = pd.DataFrame(
            columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'p-value']
        )
        
        for var in self._dependent_variables:
            aux_model = ANOVA(self._data, var, self._independent_variable, show_results=False)
            aux_model_summary = aux_model.summary()
            aux_model_summary.index = [f'{var}: {res}' for res in ['Between Groups', 'Within Groups', 'Total']]
#             aux_model_summary.index = pd.MultiIndex.from_product([[var]*3,
#                                                     ['Between Groups', 'Within Groups', 'Total']])
            summary = pd.concat([summary, aux_model_summary])
            
        return summary 