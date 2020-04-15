# Randan
_A python package for the analysis of sociological data_

### Overview
Randan is a python package that aims to provide most of the functions presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:
1. it provides the results of the analysis in a readable and understandable form, similar to SPSS
2. it gives you information about statistical significance of the parameters whenever it is possible
3. it unites all the necessary methods so you do not need to switch between different packages and software anymore

As we emphasize the importance of the way your results look like, we highly suggest to use `randan` in Jupyter Notebook.

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

### Structure
By now, **three** modules have been included in the package. These modules correspond to the SPSS functions as follows:

| Module | Class or function | Corresponding SPSS function | Description |
|--------|-------------------|-----------------------------|-------------|
| bivariate_association | Crosstab | Analyze -> Descriptive statistics -> Crosstabs | Analysis of contingency tables |
| comparison_of_central_tendency | ANOVA | Analyze -> Compare means > One-Way ANOVA | Analysis of variance |
| tree | CHAIDRegressor, CHAIDClassifier | Analyze -> Classify -> Tree -> CHAID | CHAID decision tree for scale and categorical dependent variables, respectively | 
