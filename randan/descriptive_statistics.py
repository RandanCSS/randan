import pandas as pd
import numpy as np

from pandas.api.types import is_numeric_dtype
from .utils import get_categories

from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import shapiro

from IPython.display import display

class NominalStatistics:
    """
    A class producing descriptive statistics relevant for nominal variables.

    Parameters
    ----------
    data : pd.DataFrame
        Data used to perform the analysis
    variables : list
        Variables from data to include in the analysis
    frequencies : bool
        Whether to show frequency tables
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int 
        Number of digits to round results when showing them

    """
    def __init__(
        self, 
        data, 
        variables=None,
        frequencies=True,
        show_results=True,
        n_decimals=3
    ):
        
        if variables is not None:
        
            if not isinstance(variables, list):
                phrase = 'Variables should be passed as list. Type {} was passed instead.'
                raise TypeError(phrase.format(type(variables)))
            
            else:
                self._data = data[variables].copy()
        
        else:
            self._data = data.copy()
        
        self._variables = list(self._data.columns)
        
        if show_results:
            self.show_results(n_decimals=n_decimals)
        if frequencies:
            self.show_frequencies()
    
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nNOMINAL STATISTICS SUMMARY')
        print('------------------\n')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption("method .summary()")\
                    .set_precision(n_decimals))
        if len(self._mult_modes) > 0:
            vars_ = ', '.join(self._mult_modes)
            print(f'Following variables have multiple modes: {vars_}')
    
    def show_frequencies(self, n_decimals=3):
        """
        Show frequency tables.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nFREQUENCIES')
        for var in self._variables:
            print('------------------\n')
            print(f'variable: {var}')
            display(self.frequencies()[var].style\
                        .format(None, na_rep="")\
                        .set_caption(f"method .frequencies()['{var}']")\
                        .set_precision(n_decimals))
    
    def _get_statistics(self):
        measures = {}
        self._mult_modes = []
        for var in self._variables:
            ser = self._data[var]
            n = len(ser.dropna())
            mode = ser.mode()
            if len(mode) > 1:
                self._mult_modes.append(var)
            mode = mode.iloc[0]
            entr = NominalStatistics._entropy_coef(ser)
            cqv = NominalStatistics._cqv_coef(ser)
            measures.update({var: [n, mode, entr, cqv]})
        measures = pd.DataFrame(measures, index=['N', 'mode', 'entropy coef.', 'quality var.'])
        return measures.T
    
    @staticmethod
    def _entropy_coef(series):
        p = series.value_counts(normalize=True) 
        entr_obs = (p * p.apply(np.log)).sum()
        n = len(p)
        p_exp = np.array([1/n] * n)
        entr_exp = (p_exp * np.log(p_exp)).sum()
        coef = entr_obs / entr_exp
        return coef
    
    @staticmethod
    def _cqv_coef(series):
        n = len(series)
        p = series.value_counts()
        k = len(p)
        p_sq_sum = (p ** 2).sum()
        coef = (k * (n**2 - p_sq_sum)) / ((n**2) * (k - 1))
        return coef
    
    def summary(self):
        """
        Return aggregated results of the analysis.
        """
        return self._get_statistics()
    
    def __repr__(self):
        n_vars_ = len(self._variables)
        return f'<NominalStatistics Object for {n_vars_} variables>'
    
    @staticmethod
    def _get_frequencies(series):
        raw = series.value_counts()
        raw.name = 'N'
        normalized = series.value_counts(normalize=True) * 100
        normalized.name = '%'
        return pd.concat([raw, normalized], axis=1)
    
    def frequencies(self):
        """
        Return a dictionary of all frequency tables. 
        To get a particular frequency table, use a variable's name as a key of the dictionary. 
        """
        freq = {}
        for var in self._variables:
            ser = self._data[var]
            freq.update({var: NominalStatistics._get_frequencies(ser)})
        return freq

class OrdinalStatistics:
    """
    A class producing descriptive statistics relevant for ordinal variables.

    Parameters
    ----------
    data : pd.DataFrame
        Data used to perform the analysis
    variables : list
        Variables from data to include in the analysis
    frequencies : bool
        Whether to show frequency tables
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int 
        Number of digits to round results when showing them

    """
    def __init__(
        self, 
        data, 
        variables=None,
        frequencies=True,
        show_results=True,
        n_decimals=3
    ):
        
        if variables is not None:
        
            if not isinstance(variables, list):
                phrase = 'Variables should be passed as list. Type {} was passed instead.'
                raise TypeError(phrase.format(type(variables)))
            
            else:
                self._data = data[variables].copy()
        
        else:
            self._data = data.copy()
        
        self._variables = list(self._data.columns)
        self._mappers = self._get_mappers_for_nonumerical_vars()
        
        if len(self._mappers) > 0:
            for var in self._mappers.keys():
                self._data[var] = self._data[var].map(self._mappers[var][0]).astype(float)
        
        if show_results:
            self.show_results(n_decimals=n_decimals)
        if frequencies:
            self.show_frequencies()
    
    def _get_mappers_for_nonumerical_vars(self):
        nonnum_vars = [var for var in self._variables if not is_numeric_dtype(self._data[var])]
        mappers = {}
        for var in nonnum_vars:
            mappers.update({var: OrdinalStatistics._get_mappers_for_one_var(self._data[var])})
        return mappers
    
    @staticmethod
    def _get_mappers_for_one_var(series):
        categories = get_categories(series)
        numbers = [float(num) for num in list(range(len(categories)))]
        dir_mapper = dict(zip(categories, numbers))
        inv_mapper = dict(zip(numbers, categories))
        return [dir_mapper, inv_mapper]
    
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nORDINAL STATISTICS SUMMARY')
        print('------------------\n')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption("method .summary()")\
                    .set_precision(n_decimals))
        if len(self._mult_modes) > 0:
            vars_ = ', '.join(self._mult_modes)
            print(f'Following variables have multiple modes: {vars_}')
    
    def show_frequencies(self, n_decimals=3):
        """
        Show frequency tables.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nFREQUENCIES')
        for var in self._variables:
            print('------------------\n')
            print(f'variable: {var}')
            display(self.frequencies()[var].style\
                        .format(None, na_rep="")\
                        .set_caption(f"method .frequencies()['{var}']")\
                        .set_precision(n_decimals))
    
    def _get_statistics(self):
        measures = {}
        self._mult_modes = []
        for var in self._variables:
            ser = self._data[var]
            n = len(ser.dropna())
            mode = ser.mode()
            if len(mode) > 1:
                self._mult_modes.append(var)
            mode = mode.iloc[0]
            #display(ser)
            median = ser.median()
            q25 = ser.quantile(0.25)
            q75 = ser.quantile(0.75)
            min_ = ser.min()
            max_ = ser.max()
            range_ = max_ - min_
            entr = OrdinalStatistics._entropy_coef(ser)
            cqv = OrdinalStatistics._cqv_coef(ser)
            iqv = q75 - q25
            iqv_norm = iqv / range_
            measures.update({var: [n, mode, median, q25, q75, iqv, iqv_norm, min_, max_, range_, entr, cqv]})
        measures = pd.DataFrame(measures,
                                index=['N', 'mode', 'median',
                                       '25%', '75%', 'interquart. range',
                                       'interquart. range (norm.)',
                                       'min', 'max', 'range',
                                       'entropy coef.', 'quality var.'])
        if len(self._mappers) > 0:
            for var in self._mappers.keys():
                measures.loc[['mode', 'median', '25%', '75%', 'min', 'max'], var] = \
                measures.loc[['mode', 'median', '25%', '75%', 'min', 'max'], var].map(self._mappers[var][1])
        return measures.T
    
    @staticmethod
    def _entropy_coef(series):
        p = series.value_counts(normalize=True) 
        entr_obs = (p * p.apply(np.log)).sum()
        n = len(p)
        p_exp = np.array([1/n] * n)
        entr_exp = (p_exp * np.log(p_exp)).sum()
        coef = entr_obs / entr_exp
        return coef
    
    @staticmethod
    def _cqv_coef(series):
        n = len(series)
        p = series.value_counts()
        k = len(p)
        p_sq_sum = (p ** 2).sum()
        coef = (k * (n**2 - p_sq_sum)) / ((n**2) * (k - 1))
        return coef
    
    def summary(self):
        """
        Return aggregated results of the analysis.
        """
        return self._get_statistics()
    
    def __repr__(self):
        n_vars_ = len(self._variables)
        return f'<OrdinalStatistics Object for {n_vars_} variables>'
    
    @staticmethod
    def _get_frequencies(series):
        raw = series.value_counts()
        raw.name = 'N'
        normalized = series.value_counts(normalize=True) * 100
        normalized.name = '%'
        return pd.concat([raw, normalized], axis=1)
    
    def frequencies(self):
        """
        Return a dictionary of all frequency tables. 
        To get a particular frequency table, use a variable's name as a key of the dictionary. 
        """
        freq = {}
        for var in self._variables:
            ser = self._data[var]
            freqs = OrdinalStatistics._get_frequencies(ser)
            if var in self._mappers:
                freqs.index = freqs.index.map(self._mappers[var][1])
            freq.update({var: freqs})
        return freq

class ScaleStatistics:
    """
    A class producing descriptive statistics relevant for scale (a.k.a. interval) variables.

    Parameters
    ----------
    data : pd.DataFrame
        Data used to perform the analysis
    variables : list
        Variables from data to include in the analysis
    frequencies : bool
        Whether to show frequency tables
    normality_test : bool
        Whether to perform a normality test
    normality_test_type : str
        Which normality test to use. Available values: 'ks' (Kolmogorov-Smirnov's test) or 'sw' (Shapiro-Wilk' test)
    show_results : bool 
        Whether to show results of analysis
    n_decimals : int 
        Number of digits to round results when showing them

    """
    def __init__(
        self, 
        data, 
        variables=None,
        frequencies=False,
        normality_test=False,
        normality_test_type='ks',
        show_results=True,
        n_decimals=3
    ):
        
        if variables is not None:
        
            if not isinstance(variables, list):
                phrase = 'Variables should be passed as list. Type {} was passed instead.'
                raise TypeError(phrase.format(type(variables)))
            
            else:
                self._data = data[variables].copy()
        
        else:
            self._data = data.copy()
        
        self._variables = list(self._data.columns)
        self._mappers = self._get_mappers_for_nonumerical_vars()
        
        self.normality_test_type = normality_test_type
        
        if len(self._mappers) > 0:
            for var in self._mappers.keys():
                self._data[var] = self._data[var].map(self._mappers[var][0]).astype(float)
        
        if show_results:
            if len(self._mappers)>0:
                print('\nENCODING INFORMATION')
                print('------------------\n')
                print('Some of the variables are presented as categorical ones.')
                print('They were encoded according to the following rules:')
                for var in self._mappers:
                    print('------------------\n')
                    print(f'variable: {var}')
                    display(pd.DataFrame(self._mappers[var][0], index=['Encoded value']).T)
                    
            self.show_results(n_decimals=n_decimals)
        if normality_test:
            self.show_normality_test(n_decimals=n_decimals)
        if frequencies:
            self.show_frequencies()
            
        
    
    def normality_test(self, test_type='ks'):
        """
        Perform normality tests for all included variables.
        
        Parameters
        ----------
        test_type : str
        Which normality test to use. Available values: 'ks' (Kolmogorov-Smirnov's test) or 'sw' (Shapiro-Wilk' test)
        """
        if test_type not in ['ks', 'sw']:
            raise ValueError("Unknown normality test type. Possible values: 'ks' (Kolmogorov-Smirnov) ans 'sw' (Shapiro-Wilk)")
        
        results = {}
        
        for var in self._variables:
            ser = self._data[var]
            if test_type=='ks':
                stat, pval = lilliefors(ser.dropna(), pvalmethod='approx')
            elif test_type=='sw':
                stat, pval = shapiro(ser.dropna())
            results.update({var: [stat, pval]})
        
        results = pd.DataFrame(results, index=['statistic', 'p-value'])
        
        return results.T
    
    def show_normality_test(self, n_decimals=3):
        """
        Show results of normality tests for all included variables.
        
        Parameters
        ----------
        n_decimals : int 
        Number of digits to round results when showing them
        """
        print('\nNORMALITY TESTS')
        print('------------------\n')
        display(self.normality_test(self.normality_test_type).style\
                    .format(None, na_rep="")\
                    .set_caption(f"method .normality_test(test_type='{self.normality_test_type}')")\
                    .set_precision(n_decimals))
    
    def _get_mappers_for_nonumerical_vars(self):
        nonnum_vars = [var for var in self._variables if not is_numeric_dtype(self._data[var])]
        mappers = {}
        for var in nonnum_vars:
            mappers.update({var: ScaleStatistics._get_mappers_for_one_var(self._data[var])})
        return mappers
    
    @staticmethod
    def _get_mappers_for_one_var(series):
        categories = get_categories(series)
        numbers = [float(num) for num in list(range(len(categories)))]
        dir_mapper = dict(zip(categories, numbers))
        inv_mapper = dict(zip(numbers, categories))
        return [dir_mapper, inv_mapper]
    
    def show_results(self, n_decimals=3):
        """
        Show results of the analysis in a readable form.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nSCALE STATISTICS SUMMARY')
        print('------------------\n')
        display(self.summary().style\
                    .format(None, na_rep="")\
                    .set_caption("method .summary()")\
                    .set_precision(n_decimals))
        if len(self._mult_modes) > 0:
            vars_ = ', '.join(self._mult_modes)
            print(f'Following variables have multiple modes: {vars_}')
    
    def show_frequencies(self, n_decimals=3):
        """
        Show frequency tables.
        
        Parameters
        ----------
        n_decimals : int
            Number of digits to round results when showing them
        """
        print('\nFREQUENCIES')
        for var in self._variables:
            print('------------------\n')
            print(f'variable: {var}')
            display(self.frequencies()[var].style\
                        .format(None, na_rep="")\
                        .set_caption(f"method .frequencies()['{var}']")\
                        .set_precision(n_decimals))
    
    def _get_statistics(self):
        measures = {}
        self._mult_modes = []
        for var in self._variables:
            ser = self._data[var]
            n = len(ser.dropna())
            mode = ser.mode()
            if len(mode) > 1:
                self._mult_modes.append(var)
            mode = mode.iloc[0]
            #display(ser)
            median = ser.median()
            mean = ser.mean()
            q25 = ser.quantile(0.25)
            q75 = ser.quantile(0.75)
            min_ = ser.min()
            max_ = ser.max()
            range_ = max_ - min_
            std = ser.std()
            var_ = ser.var()
            entr = ScaleStatistics._entropy_coef(ser)
            cqv = ScaleStatistics._cqv_coef(ser)
            iqv = q75 - q25
            iqv_norm = iqv / range_
            measures.update({var: [n, mode, median, mean, q25, q75, iqv, iqv_norm, min_, max_, range_, std, var_, entr, cqv]})
        measures = pd.DataFrame(measures,
                                index=['N', 'mode', 'median', 'mean',
                                       '25%', '75%', 'interquart. range',
                                       'interquart. range (norm.)',
                                       'min', 'max', 'range', 'std', 'var',
                                       'entropy coef.', 'quality var.'])
        if len(self._mappers) > 0:
            for var in self._mappers.keys():
                measures.loc[['mode', 'median', '25%', '75%', 'min', 'max'], var] = \
                measures.loc[['mode', 'median', '25%', '75%', 'min', 'max'], var].map(self._mappers[var][1])
        return measures.T
    
    @staticmethod
    def _entropy_coef(series):
        p = series.value_counts(normalize=True) 
        entr_obs = (p * p.apply(np.log)).sum()
        n = len(p)
        p_exp = np.array([1/n] * n)
        entr_exp = (p_exp * np.log(p_exp)).sum()
        coef = entr_obs / entr_exp
        return coef
    
    @staticmethod
    def _cqv_coef(series):
        n = len(series)
        p = series.value_counts()
        k = len(p)
        p_sq_sum = (p ** 2).sum()
        coef = (k * (n**2 - p_sq_sum)) / ((n**2) * (k - 1))
        return coef
    
    def summary(self):
        """
        Return aggregated results of the analysis.
        """
        return self._get_statistics()
    
    def __repr__(self):
        n_vars_ = len(self._variables)
        return f'<ScaleStatistics Object for {n_vars_} variables>'
    
    @staticmethod
    def _get_frequencies(series):
        raw = series.value_counts()
        raw.name = 'N'
        normalized = series.value_counts(normalize=True) * 100
        normalized.name = '%'
        return pd.concat([raw, normalized], axis=1)
    
    def frequencies(self):
        """
        Return a dictionary of all frequency tables. 
        To get a particular frequency table, use a variable's name as a key of the dictionary. 
        """
        freq = {}
        for var in self._variables:
            ser = self._data[var]
            freqs = OrdinalStatistics._get_frequencies(ser)
            if var in self._mappers:
                freqs.index = freqs.index.map(self._mappers[var][1])
            freq.update({var: freqs})
        return freq
