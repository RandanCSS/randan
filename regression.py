#TODO: regression without constant

import pandas as pd
import numpy as np
import scipy as sp
from statsmodels.formula.api import ols
from statsmodels.tools.tools import add_constant
from statsmodels.api import OLS
from statsmodels.stats.outliers_influence import OLSInfluence

from IPython.display import display

class LinearRegression:
    
    """
    Class for OLS regression models based on the excellent statsmodels package.
    
    Parameters:
    ----------
    method (str: 'enter', 'backward'):  
    include_constant (bool): !CURRENTLY UNAVAILIABLE! whether to include constant in the model
    sig_level_entry (float): !CURRENTLY UNAVAILIABLE! max significance level to include predictor in the model 
    sig_level_removal (float): min significance level to exclude predictor from the model
    """
    
    def __init__(
        self, 
        method='enter',
        include_constant=True,
        sig_level_entry=0.05,
        sig_level_removal=0.05
    ):
        
        self.method = method.lower().strip()
        self.include_constant = include_constant
        self.sig_level_entry = sig_level_entry
        self.sig_level_removal = sig_level_removal
        
    def fit(
        self,
        data,
        formula,
        categorical_variables=None,
        show_results=True,
        confidence_intervals=True,
        collinearity_statistics=False,
        use_patsy_notation=False,
        n_decimals=3
    ):
    
        """
        Fit model to the given data using formula.

        Parameters:
        ----------
        data (DataFrame): data to fit the model  
        formula (str): formula of a model specification, e.g. 'y ~ x1 + x2'; 
        should be passed either in Patsy (statsmodels) notation
        or using the following rules: 
        '*' for interaction of the variables,
        ':' for interaction & main effects, 
        i.e., 'y ~ x:z' equals to 'y ~ x + z + x*z' (unlike the Patsy notation).
        If you use Patsy notation, please specify the parameter use_patsy_notation=True.
        categorical_variables (list): list of names of the variables that should be considered categorical.
        These variables would be automatically converted into sets of dummy variables.
        If you want to use this option, please make sure that you don't have nested names of variables
        (e.g. 'imdb' and 'imdb_rate' at the same time), otherwise this option results in an incorrect procedure.
        show_results (bool): whether to show results of analysis
        confidence_intervals (bool): whether to include coefficients' confidence intervals in the summary table
        collinearity_statistics (bool): whether to include coefficients' tolerance and VIF in the summary table
        use_patsy_notation (bool): turn this on if you use strictly Patsy's rules to define a formula.
        See more: https://patsy.readthedocs.io/en/latest/quickstart.html
        n_decimals (int): number of digits to round results when showing them
        """

        self._data = data.copy()
        
        self.categorical_variables = categorical_variables
        self._show_ci = confidence_intervals
        self._show_col = collinearity_statistics
        
        if '=' in formula:
            formula = formula.replace('=', '~')
        
        if not use_patsy_notation:
            formula = formula.replace('*', '^').replace(':', '*').replace('^', ':')
            
        self.formula = formula
        
        #won't work correctly if some variables have similar names (e.g. kinopoisk_rate and kinopoisk_rate_count)
        if categorical_variables is not None:
            if not isinstance(categorical_variables, list):
                raise ValueError(f"""Categorical variables should be passed as list.
                Type {type(categorical_variables)} was passed instead.""")
            else:
                for variable in categorical_variables:
                    formula = formula.replace(variable, f'C({variable})')
                    
        self._model = ols(formula=formula, data=data).fit()
        self._observations_idx = list(self._model.fittedvalues.index)
        self.dependent_variable = self._model.model.endog_names        
        self.variables_excluded = self._identify_variables_without_variation()
        
        if len(self.variables_excluded) > 0:
            y = pd.Series(self._model.model.endog.copy(), 
                          index=self._observations_idx,
                          name=self.dependent_variable)
            X = self._remove_variables_without_variation()
            self._model = OLS(y, X, missing = 'drop').fit()
            self.variables_excluded = [LinearRegression._translate_from_patsy_notation(x) for x in self.variables_excluded]
        
        if self.method == 'backward':
            self._fit_backward()
        
        self._get_statistics_from_model()
        
        self.predictions = self.predict()
        
        if show_results:
            self.show_results(n_decimals)
        
        if len(self.variables_excluded) > 0:
            print('------------------\n')
            print(f"Following variables were excluded due to zero variance: {'; '.join(self.variables_excluded)}")
        
        return self
    
    def predict(
        self,
        data=None,
        add_to_data=False,
    ):
        """
        Predict values of a dependent variable using a fitted model.
        
        Parameters:
        ----------
        data (DataFrame): data for prediction; 
        may be not specified if you want to predict values for the same data that were used to fit a model
        add_to_data (bool): whether to merge predictions with the given data.
        Currently, this option returns data with a sorted index
        """
        name = f'{self.dependent_variable} (predicted)'
        
        if data is None:
            data_init = self._data.copy()
            result = self._model.fittedvalues
            data_init[name] = result
            if add_to_data:
                return data_init
            else:
                return data_init[name].copy()
                  
        else:
            aux_model = ols(self.formula, data).fit()
            aux_data_idx = aux_model.fittedvalues.index
            aux_data_cols = aux_model.model.exog_names
            aux_data_cols = [LinearRegression._translate_from_patsy_notation(x)\
                              for x in aux_data_cols]
            aux_data = pd.DataFrame(aux_model.model.exog,
                                    index=aux_data_idx,
                                    columns=aux_data_cols)
            aux_X = add_constant(aux_data[self.variables_included].copy())
            aux_y = aux_model.model.endog.copy()
                  
            aux_model = OLS(aux_y, aux_X, missing='drop').fit()
            result = aux_model.fittedvalues
            result.name = name      
            if add_to_data:
                result = pd.concat([data, result], axis=1, sort=False)
                  
            return result
        
    
    def _get_statistics_from_model(self):
        
        self.N = self._model.nobs
        self.r2 = self._model.rsquared
        self.r2_adjusted = self._model.rsquared_adj
        self.F = self._model.fvalue
        self.F_pvalue = self._model.f_pvalue
        self.ess = self._model.ess
        self.rss = self._model.ssr
        if self.include_constant:
            self.tss = self._model.centered_tss
        else:
            self.tss = self._model.uncentered_tss
        self.ms_model = self._model.mse_model
        self.ms_resid = self._model.mse_resid
        self.ms_total = self._model.mse_total
        self.dof_model = self._model.df_model
        self.dof_resid = self._model.df_resid
        self.dof_total = self.dof_model + self.dof_resid
        
        self.coefficients = self._model.params.copy()
        self.coefficients_sterrors = self._model.bse.copy()
        self.coefficients_tvalues = self._model.tvalues.copy()
        self.coefficients_pvalues = self._model.pvalues.copy()
        
        variables_included = [x for x in list(self.coefficients.index) if x!='Intercept']
        self._variables_included_patsy = variables_included.copy()
        
        variables_included = [LinearRegression._translate_from_patsy_notation(x) for x in variables_included]        
        
        self.variables_included = variables_included
        #self._independent_variables = 
        
        if self.include_constant:
            self._params_idx = ['Constant'] + variables_included
        else:
            self._params_idx = variables_included.copy()
        
        for stats in [self.coefficients,
                     self.coefficients_pvalues,
                     self.coefficients_sterrors,
                     self.coefficients_tvalues]:
            stats.index = self._params_idx    
        
        return
    
    @property
    def coefficients_beta(self):
        b = np.array(self._model.params)[1:]
        std_y = self._model.model.endog.std(axis=0)
        std_x = self._model.model.exog.std(axis=0)[1:]
        beta = list(b * (std_x / std_y))
        
        if self.include_constant:
            beta = [np.nan] + beta
            
        result = pd.Series(beta, index=self._params_idx)
        return result
    
    @property
    def coefficients_confidence_interval(self):
        
        ci = self._model.conf_int()
        ci.index = self._params_idx
        
        ci.columns = [f'LB CI (95%)',
                     f'UB CI (95%)']
        return ci
    
    @property
    def coefficients_VIF(self):
        #eps = 1e-20
        x = self._model.model.exog[:, 1:].copy()
        inv_corr = np.linalg.inv(sp.corrcoef(x, rowvar=False))
        diag = list(inv_corr.diagonal())
        if self.include_constant:
            diag = [np.nan] + diag
        
        return pd.Series(diag, index=self._params_idx)
    
    @property
    def coefficients_tolerance(self):
        return 1 / self.coefficients_VIF
    
    @staticmethod
    def _translate_from_patsy_notation(effect):
        effect = effect\
        .replace(':', ' * ')\
        .replace('C(', '')\
        .replace('T.', '')\
        .replace('[', ' = "')\
        .replace(']', '"')\
        .replace(')', '')
        
        return effect
    
    def show_results(self, n_decimals):
        """
        Show results of the analysis in a readable form.
        
        Parameters:
        ----------
        n_decimals (int): number of digits to round results when showing them
        """
        phrase = 'method {}'
        
        print('\nLINEAR REGRESSION SUMMARY')
        print('------------------\n')
        print('Model summary')
        display(self.summary_r2().style\
                    .set_caption(phrase.format('.summary_r2()'))\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('ANOVA')
        display(self.summary_F().style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.summary_F()'))\
                    .set_precision(n_decimals))
        print('------------------\n')
        print('Coefficients')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption(phrase.format('.summary()'))\
                    .set_precision(n_decimals))
    
    def summary(self):
        """
        Summary table with requested information related to regression coefficients.
        """
                  
        statistics = [
            self.coefficients,
            self.coefficients_sterrors,
            self.coefficients_beta,
            self.coefficients_tvalues,
            self.coefficients_pvalues
        ]
        
        columns = [
            'B', 
            'Std. Error',
            'Beta',
            't',
            'p-value'
        ]
        
        if self._show_ci:
            statistics.append(self.coefficients_confidence_interval)
            columns.extend(list(self.coefficients_confidence_interval.columns))
            
        if self._show_col:
            statistics.append(self.coefficients_tolerance)
            statistics.append(self.coefficients_VIF)
            columns.extend(['Tolerance', 'VIF'])
        
        statistics = pd.concat(statistics, axis=1)
        
        statistics.columns = columns
        
        statistics.index = self._params_idx
        
        return statistics
    
    def summary_r2(self):
        """
        Summary table with information related to coefficient of determination.
        """
        r = self.r2 ** 0.5
        r2 = self.r2
        r2_adj = self.r2_adjusted
        
        statistics = [[r, r2, r2_adj]]
        columns = [
            'R',
            'R Squared',
            'Adj. R Squared'
        ]
        
        statistics = pd.DataFrame(
            statistics,
            columns=columns,
            index = ['']
        )
        
        return statistics
    
    def summary_F(self):
        """
        Summary table with information related to F-statistic.
        """
        
        results = [[self.ess, self.dof_model, self.ms_model, self.F, self.F_pvalue],
                  [self.rss, self.dof_resid, self.ms_resid, np.nan, np.nan],
                  [self.tss, self.dof_total, np.nan, np.nan, np.nan]]
        
        results = pd.DataFrame(results,
                              columns = ['Sum of Squares', 'df', 'Mean Square', 'F', 'p-value'],
                              index = ['Regression', 'Residual', 'Total'])
                
        return results
    
    def _fit_backward(self):
        
        y_train = pd.Series(self._model.model.endog.copy(), 
                            name=self.dependent_variable,
                           index=self._observations_idx)
        X_train = pd.DataFrame(self._model.model.exog, columns=self._model.model.exog_names,
                              index=self._observations_idx)

        model = OLS(y_train, X_train, missing = 'drop')
        
        results = model.fit()
        
        max_pvalue = results.pvalues.drop('Intercept').max()
        
        while max_pvalue > self.sig_level_removal:
            x_to_drop = results.pvalues.drop('Intercept').idxmax()
            X_train = X_train.drop(x_to_drop, axis = 1)
            model = OLS(y_train, X_train, missing = 'drop')
            results = model.fit()
            max_pvalue = results.pvalues.drop('Intercept').max()
        
        self._model = results
        
        return
    
    def _identify_variables_without_variation(self):
        if self.include_constant:
            mask = self._model.model.exog.var(axis=0)[1:] == 0
        else:
            mask = self._model.model.exog.var(axis=0) == 0
            
        variables_included = [x for x in list(self._model.params.index) if x!='Intercept']

        return list(np.array(variables_included)[mask])
    
    def _remove_variables_without_variation(self):
        X = pd.DataFrame(self._model.model.exog, 
                         columns=self._model.model.exog_names, 
                        index=self._observations_idx)
        X = X.drop(self.variables_excluded, axis = 1)
        return X
                  
    def save_independent_variables(
        self,
        data=None,
        add_to_data=False
    ):
        """
        Produce values of independent variable remained in a fitted model.
        This option is useful if you don't create dummy variables or interaction effects manually
        but want to use them in a further analysis. Only variables remained in a model are returned
        (those that are shown in a summary table).
        
        Parameters:
        ----------
        data (DataFrame): data for which independent variables are requested; 
        may be not specified if you want to save values for the same data that were used to fit a model
        add_to_data (bool): whether to merge new values with the given data.
        Currently, this option returns data with a sorted index
        """
                  
        if data is None:
            data = self._data.copy()
            if self.include_constant:
                result = self._model.model.exog[:, 1:].copy()
            else:
                result = self._model.model.exog.copy()
            columns = [x for x in self.variables_included if x!='Constant']
            result = pd.DataFrame(
                result,
                columns=columns,
                index=self._observations_idx)
        
        else:
            aux_model = ols(self.formula, data).fit()
            aux_data_idx = aux_model.fittedvalues.index
            aux_data_cols = aux_model.model.exog_names
            aux_data_cols = [LinearRegression._translate_from_patsy_notation(x)\
                              for x in aux_data_cols]
            aux_data = pd.DataFrame(aux_model.model.exog,
                                    index=aux_data_idx,
                                    columns=aux_data_cols)
            result = aux_data[self.variables_included]
        
        if add_to_data:
            result = pd.concat([data, result], axis=1, sort=False)
                  
        return result
            
    def save_residuals(self,
                      unstandardized=True,
                      standardized=False,
                      studentized=False,
                      deleted=False,
                      studentized_deleted=False,
                      add_to_data=False):
        """
        Produce values of various residuals. Residuals are returned only for data used to fit a model.
        
        Parameters:
        ----------
        unstandardized (bool): whether to save unstandardized (raw) residuals
        standardized (bool): whether to save standardized (z-scores) residuals
        studentized (bool): whether to save studentized residuals
        deleted (bool): whether to save deleted residuals
        studentized_deleted (bool): whether to save studentized deleted residuals
        add_to_data (bool): whether to merge new values with data.
        Currently, this option returns data with a sorted index
        """
                  
        columns_to_show = [f'{k.capitalize().replace("ized", ".").replace("eted", ".").replace("_", " ")} res.' \
                           for k, v in vars().items() if v==True and k!='add_to_data']
        
        infl = OLSInfluence(self._model)
        
        result = []
                  
        res_unstand = infl.resid
        res_unstand.name = 'Unstandard. res.'
                  
        res_stand = (res_unstand - res_unstand.mean()) / res_unstand.std()
        res_stand.name = 'Standard. res.'
                  
        res_stud = infl.resid_studentized_internal
        res_stud.name = 'Student. res.'
        
        result.extend([
            res_unstand, 
            res_stand,
            res_stud])
        
        if deleted:
            res_del = infl.resid_press
            res_del.name = 'Del. res.'
            result.append(res_del)                                                       
                           
        if studentized_deleted:
            res_stud_del = infl.resid_studentized_external
            res_stud_del.name = 'Student. del. res.'
            result.append(res_stud_del)      

        result = pd.concat(result, axis=1)
        result = result[columns_to_show].copy()
            
        if add_to_data:
            result = pd.concat([self._data, result], axis=1)
            
        return result
                           
    #following two methods are still in progress                       
    @staticmethod
    def _turn_all_rows_to_fancy(summary):
        return summary.apply(lambda x: LinearRegression._turn_one_row_to_fancy(x), axis=1)
    
    @staticmethod
    def _turn_one_row_to_fancy(row):
        coef = round(row['B'].item(), 3)
        sterr = round(row['Std. Error'].item(), 3)
        pval = row['p-value'].item()
        
        if pval <= 0.01:
            mark = '***'
        elif pval <= 0.05:
            mark = '**'
        elif pval <= 0.1:
            mark = '*'
        else:
            mark = ''
                           
        result = f'{coef}{mark} \n ({sterr})'
        return result