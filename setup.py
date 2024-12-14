import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randan", # Replace with your own username
    version="0.3",
    author="Aleksei Rotmistrov, Svetlana Zhuchkova",
    author_email="alexey.n.rotmistrov@gmail.com, lana_lob@mail.ru",
    description="A python package for the analysis of social data",
    long_description=long_description,
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
        'graphviz>=0.14'
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
