import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randan", # Replace with your own username
    version="0.3.0",
    download_url='https://github.com/RandanCSS/randan/archive/refs/tags/v0.3.0.tar.gz',
    author="Aleksei Rotmistrov, Svetlana Zhuchkova",
    author_email="alexey.n.rotmistrov@gmail.com, lana_lob@mail.ru",
    description="A python package for gaining social and financial data and their analysis",
    long_description='randan is a python package that aims to help social scientists, statisticians and financier. For the former ones it provides twelve analytical modules that emulate the most popular options presented in SPSS. Unlike the other python packages for data analysis, it has three main features, which make it attractive for social scientists:
        1. it provides the results of the analysis in a readable and understandable form, similar to SPSS
        2. it provides information about statistical significance of the parameters whenever possible
        3. it meets the most popular analytical needs of social scientists; so, the switching among different packages and software stays in the past
        A new -- thirteenth -- module provides data from YouTube literally by couple of clicks.',
    long_description_content_type="text/markdown",
    url="https://github.com/RandanCSS/randan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'factor_analyzer',
        'google-api-python-client>=2.154.0',
        'graphviz>=0.14',
        'matplotlib>=3.1.1',
        'numpy>=1.18.4',
        'pandas>=1.0.3',
        'pydot>=1.4.1',
        'regex>=2024.9.11',
        'scikit-learn>=0.23.1',
        'statsmodels>=0.11.1',
        'tqdm>=4.66.5',
    ],
    python_requires='>=3.6',
)
