.. randan documentation master file, created by
   sphinx-quickstart on Sat May 30 17:59:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

randan documentation
==================================

Overview
==================
``randan`` is a python package that aims to provide most of the functions presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:

1. it provides the results of the analysis in a readable and understandable form, similar to SPSS
2. it gives you information about statistical significance of the parameters whenever possible
3. it unites all the necessary methods so you do not need to switch between different packages and software anymore

``randan`` should be used in Jupyter Notebook and your data should be stored in ``pandas`` DataFrames.

This documentation aims to provide information about how ``randan``'s classes are organized.
For examples of usage and detailed tutorials, go to `Github page <https://github.com/LanaLob/randan>`_ of this project.

Use the links below to explore ``randan``'s functionality.

.. toctree::
   :maxdepth: 2
   :caption: Univariate analysis:

   descriptive_statistics

.. toctree::
   :maxdepth: 2
   :caption: Bivariate analysis:

   bivariate_association
   comparison_of_central_tendency

.. toctree::
   :maxdepth: 2
   :caption: Multivariate analysis:

   clustering
   dimension_reduction
   regression
   tree

   autoapi/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note:: This project is still under development but you can use all the modules presented in this documentation.

