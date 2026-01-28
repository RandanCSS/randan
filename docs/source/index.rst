.. randan documentation master file, created by
   sphinx-quickstart on Sat May 30 17:59:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

randan documentation
==================================

Overview
==================
randan is a python package that aims to help social scientists, statisticians and financiers. For the former ones it provides twelve analytical modules that emulate the most popular options presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:

- it provides the results of the analysis in a readable and understandable form, similar to SPSS
- it provides information about statistical significance of the parameters whenever possible
- it meets the most popular analytical needs of social scientists; so, the switching among different packages and software stays in the past

New -- thirteenth and fourteenth -- modules provide data from YouTube and VK respectively literally by couple of clicks.

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

