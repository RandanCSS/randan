import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randan", # Replace with your own username
    version="0.1.2",
    author="Svetlana Zhuchkova",
    author_email="lana_lob@mail.ru",
    description="A python package for the analysis of social data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LanaLob/randan",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.4',
        'pandas>=1.0.3',
        'scikit-learn>=0.23.1',
        'statsmodels>=0.11.1',
        'matplotlib>=3.1.1',
        'factor_analyzer',
        'pydot>=1.4.1',
        'graphviz>=0.14'
    ],
    python_requires='>=3.6',
)
