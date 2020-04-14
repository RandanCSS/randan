import pandas as pd
import numpy as np
from scipy.stats import f
from .utils import get_categories
from IPython.display import display

class ANOVA:
    """
    Class to perform analysis of variance (AnOVa).

    Parameters:
    ----------
    data (DataFrame): data used to perform the analysis
    dependent_variable (str): name of a scale dependent variable
    independent_variable (str): name of a independent (factor, grouping) variable
    show_results (bool): whether to show results of analysis
    n_decimals (int): number of digits to round results when showing them
    """
    def __init__(self, 
                 data, 
                 dependent_variable, 
                 independent_variable,
                 show_results=True,
                 n_decimals=3):
        
        data = data[[dependent_variable, independent_variable]].dropna()
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
        data['mean'] = groups_means[dependent_variable]['mean']

        SSw = ((data[dependent_variable] - data['mean'])**2).sum()
        
        if dof_w > 0:
            MSw = SSw / dof_w
        else:
            MSw = np.nan
        
        groups_means['grand_mean'] = data[dependent_variable].mean()

        SSb = (((groups_means[dependent_variable]['mean'] - groups_means['grand_mean'])**2)*groups_means[dependent_variable]['count']).sum()
        MSb = SSb / dof_b
        
        SSt = SSw + SSb
        if pd.isnull(MSw):
            F = 0
            pvalue = 1
        else:
            F = MSb / MSw
            pvalue = f.sf(F, dof_b, dof_w)
        
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
        
        self._dependent_variable = dependent_variable
        self._independent_variable = independent_variable
        
        if show_results:
            self.show_results(n_decimals=n_decimals)
        
    def summary(self):
        """
        Get summary information on the conducted analysis.
        """
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
        
        Parameters:
        ----------
        n_decimals (int): number of digits to round results when showing them
        """

        
        print('\nANOVA SUMMARY')
        print('------------------')
        display(self.summary().style\
                .format(None, na_rep="")\
                .set_caption(f"""Dependent variable: {self._dependent_variable},
                independent variable: {self._independent_variable}""")\
                .set_precision(n_decimals))