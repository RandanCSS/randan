# Randan
_A python package for the analysis of sociological data_

### Overview
Randan is a python package that aims to provide most of the functions presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:
1. it provides the results of the analysis in a readable and understandable form, similar to SPSS
2. it gives you information about statistical significance of the parameters whenever it is possible
3. it unites all the necessary methods so you do not need to switch between different packages and software anymore

As we emphasize the importance of the way your results look like, we highly suggest to use `randan` in Jupyter Notebook and store your data in `pandas` DataFrames.

> _**N.B.:** You should understand that this project is under development now, which means it is constantly updating. We will have finished the main part of the package by the middle of June, 2020._ 

### Installation
Currently, `randan` is **unavailable** to install by `pip`. However, you can use two options to download our package:

1. download it manually by using `Clone or download` -> `Download ZIP` and unzip the downloaded archive in a desired folder
2. download it automatically by running `!git clone https://github.com/LanaLob/randan.git` in Jupyter Notebook. This will create a local copy of the current version of our package on your computer. To specify any particular folder for the package, modify this command as follows: `!git clone https://github.com/LanaLob/randan.git [desired path]`. If you have troubles with `git`, try to install it from the [official source](https://git-scm.com/downloads) (you likely have to restart your computer afterwards).

We recommend using the second way and clone our package every time you're going to use it as it allows you to have all the updates in time.

Once you have the package in the folder, you can import it as any python package:

```python
# like this
import randan

# or like this
from randan.tree import CHAIDRegressor

# etc.
```

**Important note**: `randan` depends on the several packages, most of which are considered pre-built packages (this means you likely do not have to install them manually). However, there are two dependencies that are still required to install if you're going to use `randan.tree` module. If so, please install them by running this command in any command-line interface (such as Terminal on MacOS, cmd on Windows, Anaconda Prompt on both etc.):

```
conda install graphviz pydot
```

### Structure
By now, **three** modules have been included in the package. These modules correspond to the SPSS functions as follows:

| Module | Class or function | Corresponding SPSS function | Description |
|--------|-------------------|-----------------------------|-------------|
| bivariate_association | Crosstab | Analyze -> Descriptive statistics -> Crosstabs | Analysis of contingency tables |
| comparison_of_central_tendency | ANOVA | Analyze -> Compare means > One-Way ANOVA | Analysis of variance |
| tree | CHAIDRegressor, CHAIDClassifier | Analyze -> Classify -> Tree -> CHAID | CHAID decision tree for scale and categorical dependent variables, respectively | 

### Quick start
Although `randan` is built to be similar to SPSS, it reproduces the fit-predict and fit-transform approach, which is now being used in the most popular machine learning python packages. This approach means that you should, firstly, initialize your model and then, secondly, fit it to your data (i.e., use the `fit` function) if necessary. 
> 1. If the method you use belongs to the *unsupervised methods* (i.e., you *do not have* a dependent variable in your data), you can then use `transform` function to get values of the obtained, hidden, dependent variable such as cluster membership, factor scores etc. 
> 2. If the method you use belongs to the *supervised methods* (i.e., you *have* a dependent variable in your data), you can then use `predict` function to get values of the given dependent variable. 
> 3. If the method does not assume to estimate new values for your data (such methods are crosstabs, t-tests etc.), then it does not require to use `fit` and `transform` / `predict` functions. 

If you want to see the full list of the availiable functions associated with some class, literally ask for the help:
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
anv = ANOVA(data, dependent_variable='kinopoisk_rate', independent_variable='genre')

# however, if you want to somehow use separate statistics, you can call them this way
print(anv.F, anv.pvalue, anv.SSt)
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
    scale_variables=['year', 'time', 'kinopoisk_rate_count']
    )

#this is how you can predict values of the dependent variable and the node membership for the given data 
predictions = chaid.predict(data, node=True)
```
