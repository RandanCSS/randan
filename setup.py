from setuptools import setup, find_packages
import os

name="randan"
version="0.3.1.9"
slash = '\\' if os.name == 'nt' else '/' # выбор слэша в зависимости от ОС

folderS = []
for item in os.listdir(name):
    if '.' not in item: folderS.append(item)

folderFileS = []
for folder in folderS:
    for file in os.listdir(name + slash + folder):
        folderFileS.append(folder + '/' + file)

setup(
    name=name,
    version=version,
    author="Aleksei Rotmistrov, Svetlana Zhuchkova",
    author_email="alexey.n.rotmistrov@gmail.com, lana_lob@mail.ru",
    package_data={'randan': folderFileS},
    url="https://github.com/RandanCSS/randan",
    license="MIT",
    description="A python package for gaining social and financial data and their analysis",
    long_description=open('README.markdown').read() if os.path.exists("README.markdown") else "",
    download_url=f'https://github.com/RandanCSS/randan/archive/refs/tags/v{version}.tar.gz',
    keywords=["data", "analysis", "spss", "youtube"],
    
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
        'vk_requests>=1.2.1',
    ],
    python_requires='>=3.6',
)
