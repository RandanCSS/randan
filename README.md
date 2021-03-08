# Randan
_A python package for the analysis of social data_

_Current version: 0.2.1_

_Documentation: https://randan.readthedocs.io/en/latest/_

_If you want to contribute or report a bug, do not hesitate to [open an issue](https://github.com/LanaLob/randan/issues) on this page or contact us: alexey.n.rotmistrov@gmail.com (Aleksei Rotmistrov), lana_lob@mail.ru (Svetlana Zhuchkova)._ 

### Overview
Randan is a python package that aims to provide most of the functions presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:
1. it provides the results of the analysis in a readable and understandable form, similar to SPSS
2. it gives you information about statistical significance of the parameters whenever possible
3. it unites all the necessary methods so you do not need to switch between different packages and software anymore

As we emphasize the importance of the way your results look like, we highly suggest to use `randan` in Jupyter Notebook and store your data in `pandas` DataFrames.

> _**N.B.:** You should understand that this project is under development now, which means it is constantly updating. But you can use all the modules and classes presented in the last release._ 

### Installation
You can easily install the package from the PyPi by running:

```
pip install randan
```
If something goes wrong during the installation, consider using this code:

```
pip install --user randan
```

To upgrade package's version, run this code:
```
pip install --upgrade randan
```

Once you install the package, you can import it as any python package:

```python
# like this
import randan

# or like this
from randan.tree import CHAIDRegressor

# etc.
```

### Structure
By now, **seven** modules have been included in the package. These modules correspond to the SPSS functions as follows:

| Module | Class or function | Corresponding SPSS function | Description |
|--------|-------------------|-----------------------------|-------------|
| descriptive_statistics | NominalStatistics | Analyze -> Descriptive statistics -> Frequencies, Descriptives, Explore | Descriptive statistics relevant for nominal variables |
| descriptive_statistics | OrdinalStatistics | Analyze -> Descriptive statistics -> Frequencies, Descriptives, Explore | Descriptive statistics relevant for ordinal variables |
| descriptive_statistics | ScaleStatistics | Analyze -> Descriptive statistics -> Frequencies, Descriptives, Explore | Descriptive statistics relevant for scale (interval) variables |
| bivariate_association | Crosstab | Analyze -> Descriptive statistics -> Crosstabs | Analysis of contingency tables |
| bivariate_association | Correlation | Analyze -> Correlate -> Bivariate | Correlation coefficients |
| comparison_of_central_tendency | ANOVA | Analyze -> Compare means -> One-Way ANOVA | Analysis of variance |
| clustering | KMeans | Analyze -> Classify -> K-Means Cluster | Cluster analysis with k-means algorithm |
| dimension_reduction | CA | Analyze -> Dimension Reduction -> Correspondence Analysis | Correspondence analysis |
| dimension_reduction | PCA | Analyze -> Dimension Reduction -> Factor (extraction method: principal components) | Principal component analysis |
| regression | LinearRegression | Analyze -> Regression -> Linear | OLS regression |
| regression | BinaryLogisticRegression | Analyze -> Regression -> Binary Logistic | Binary logistic regression |
| tree | CHAIDRegressor, CHAIDClassifier | Analyze -> Classify -> Tree -> CHAID | CHAID decision tree for scale and categorical dependent variables, respectively |

### Quick start
Although `randan` is built to be similar to SPSS, it reproduces the fit-predict and fit-transform approach, which is now being used in the most popular machine learning python packages. This approach means that you should, firstly, initialize your model and then, secondly, fit it to your data (i.e., use the `fit` function) if necessary. 
> 1. If the method you use belongs to the *unsupervised methods* (i.e., you *do not have* a dependent variable in your data), you can then use `transform` function to get values of the obtained, hidden, dependent variable such as cluster membership, factor scores etc. 
> 2. If the method you use belongs to the *supervised methods* (i.e., you *have* a dependent variable in your data), you can then use `predict` function to get values of the given dependent variable. 
> 3. If the method does not assume to estimate new values for your data (such methods are crosstabs, t-tests etc.), then it does not require to use `fit` and `transform` / `predict` functions. 

If you want to see the full list of the availiable functions associated with some class, please visit our [documentation page](https://randan.readthedocs.io/en/latest/) or literally ask for help:
```python
from randan.bivariate_association import Crosstab
help(Crosstab)
```

#### Module `bivariate_association`
This module aggregates methods devoted to searching for statistical relationships between two variables. These methods do not require to use `fit` function, i.e. you only need to call the necessary class:
```python
from randan.bivariate_association import Crosstab

# with this code, you will immediately see the results
ctab = Crosstab(data, row='genre', column='age_ord')

# however, if you want to somehow use separate statistics, you can call them this way
print(ctab.chi_square, ctab.pvalue, ctab.n_cells)
```

#### Module `comparison_of_central_tendency`
This module contains both parametric and non-parametric methods for comparison of central tendency statistics. These methods do not require to use `fit` function, i.e. you only need to call the necessary class:
```python
from randan.comparison_of_central_tendency import ANOVA

# with this code, you will immediately see the results
anv = ANOVA(data, dependent_variables='kinopoisk_rate', independent_variable='genre')

# however, if you want to somehow use separate statistics, you can call them this way
print(anv.F, anv.pvalue, anv.SSt)
```
#### Module `clustering`
This module includes two main clustering methods: k-means and hierarchical (agglomerative) clustering. 

Clustering methods belong to unsupervised learning, which means you should use the `fit` function after calling the appropriate class, and then, if necessary, the `transform` function to acquire cluster membership (and / or distances to each center in case of k-means).
```python
from randan.clustering import KMeans

# with this code, you will immediately see the results, including visualization of clusters
km = KMeans(2).fit(data, ['year', 'time', 'kinopoisk_rate_count'])

# this is how you can predict the cluster membership, 
# and the distances from each observation to each cluster's center
clusters = km.transform(distance_to_centers=True)
```
> If you experience troubles with visualization and see captions like <Figure size 800x500 with 1 Axes> instead of plots, just re-run the code that produces them.
#### Module `dimension_reduction`
This module unites methods for factorization of nominal and scale variables: correspondence analysis (class `CA`) and principal component analysis (class `PCA`). 

Factorization methods belong to unsupervised learning, which means you should use the `fit` function after calling the appropriate class, and then, if necessary, the `transform` function to acquire so-called factor scores.
```python
from randan.dimension_reduction import PCA 
 
vars_ = ['trstprl', 'trstlgl', 'trstplc', 'ppltrst', 'pplfair', 'pplhlp']

# with this code, you will immediately see the results
pca = PCA(n_components=2, rotation='varimax').fit(data, variables=vars_)

# this is how you can predict the factor scores
f_scores = pca.transform()
```
#### Module `regression`
This module consists of two classical regression models: linear regression and binary logistic regression. This group of methods belongs to supervised learning, which means you should use the `fit` function after calling the appropriate class, and then, if necessary, the `predict` function to acquire predictions.
```python
from randan.regression import LinearRegression

# with this code, you will immediately see the results
formula = 'kinopoisk_rate = time + year + genre + genre*type'

regr = LinearRegression().fit(
    data, 
    formula=formula,
    categorical_variables=['genre', 'type'],
    collinearity_statistics=True
)

# this is how you can predict values of the dependent variable for the given data... 
predictions = regr.predict()

# ... save various types of residuals ...
residuals = regr.save_residuals(unstardandized=False, studentized=True)

# ... and even save values of independent variables 
# if you didn't create them manually (e.g. dummies and interactions) ...
indep_vars = regr.save_independent_variables()
```
#### Module `tree`
This module includes various methods of building decision trees. If you have a categorical dependent variable, please use those methods that contain `Classifier` part in their names. Otherwise, if you have a scale dependent variable, please use the methods that contain `Regressor` part in their names.

This group of methods belongs to supervised learning, which means you should use the `fit` function after calling the appropriate class, and then, if necessary, the `predict` function to acquire predictions.
```python
from randan.tree import CHAIDRegressor

# with this code, you will immediately see the results, including the plot of your tree
chaid = CHAIDRegressor().fit(
    data,
    dependent_variable='kinopoisk_rate',
    independent_variables=['genre', 'age_ord', 'year', 'time', 'type', 'kinopoisk_rate_count'],
    scale_variables=['year', 'time', 'kinopoisk_rate_count'],
    ordinal_variables=['age_ord']
    )

# this is how you can predict values of the dependent variable, the node membership, 
# and the description of the node in terms of interactions for the given data 
predictions = chaid.predict(node=True, interaction=True)
```
