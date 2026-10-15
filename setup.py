from setuptools import setup, find_packages
import os

name = "randan"
version = "1.5.19"
slash = '\\' if os.name == 'nt' else '/'  # выбор слэша в зависимости от ОС

if not os.path.isdir(name): folderS = []
for item in os.listdir(name):
    if '.' not in item:
        folderS.append(item)

folderFileS = []
for folder in folderS:
    for file in os.listdir(name + slash + folder):
        folderFileS.append(folder + '/' + file)

setup(
    author="Aleksei Rotmistrov, Svetlana Zhuchkova",
    author_email="alexey.n.rotmistrov@gmail.com, lana_lob@mail.ru",
    description="A python package for gaining social and financial data and their analysis",
    download_url=f'https://github.com/RandanCSS/randan/archive/refs/tags/v{version}.tar.gz',
    keywords=["data", "analysis", "spss", "youtube"],
    license="MIT",
    long_description=open('README.markdown').read() if os.path.exists("README.markdown") else "",
    name=name,
    package_data={'randan': folderFileS},
    packages=find_packages(include=[name, f'{name}.*']),
    url="https://github.com/RandanCSS/randan",
    version=version,

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # --- веб-автоматизация и парсинг ---
        'beautifulsoup4>=4.12.3',
        'requests>=2.32.3',
        'selenium>=4.27.1',
        'undetected-chromedriver>=3.5.5',
        'urllib3>=2.2.0',

        # --- визуализация и графики ---
        'graphviz>=0.21',
        'matplotlib>=3.1.1',
        'pydot>=4.0.1',

        # --- работа с Google API и gRPC ---
        'google-api-python-client>=2.154.0',
        'googledrivedownloader>=0.4',
        'grpcio-tools>=1.80.0',
        'protobuf>=5.29.1',

        # --- статистика и анализ данных ---
        'factor_analyzer>=0.5.1',
        'numpy>=1.18.4',
        'pandas>=1.0.3',
        'scikit-learn>=0.23.1',
        'statsmodels>=0.15.0',

        # --- текстовые и языковые утилиты ---
        'autocorrect>=2.6.1',
        'pymystem3>=0.2.0',
        'regex>=2024.9.11',
        'stop-words>=2025.11.4',

        # --- Excel и вспомогательные утилиты ---
        'ipython>=7.0.0',
        'openpyxl>=3.1.5',
        'tqdm>=4.66.5',
        'xlsxwriter>=3.2.9'
    ],
    python_requires='>=3.9',
)
