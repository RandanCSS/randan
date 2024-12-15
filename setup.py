import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randan", # Replace with your own username
    packages=["randan"],
    version="0.3.1.2",
    license="MIT",
    description="A python package for gaining social and financial data and their analysis",
    author="Aleksei Rotmistrov, Svetlana Zhuchkova",
    author_email="alexey.n.rotmistrov@gmail.com, lana_lob@mail.ru",
    url="https://github.com/RandanCSS/randan",
    download_url='https://github.com/RandanCSS/randan/archive/refs/tags/v0.3.1.2.tar.gz',
    keywords=["data", "analysis", "spss", "youtube"],
    # long_description='randan is a python package that aims to help social scientists, statisticians and financiers.'
    #     , 'For the former ones it provides twelve analytical modules that emulate the most popular options presented in SPSS.'
    #     , 'A new -- thirteenth -- module provides data from YouTube literally by couple of clicks.',
    # long_description_content_type="text/markdown",
    # packages=setuptools.find_packages(),
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
