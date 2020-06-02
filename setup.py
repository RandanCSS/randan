from distutils.core import setup
setup(
  name = 'randan',
  packages = ['randan'],
  version = '0.1',
  license='MIT',
  description = 'A python package for the analysis of social data',
  author = 'Svetlana Zhuchkova',
  author_email = 'lana_lob@mail.ru',
  url = 'https://github.com/LanaLob/randan',
  download_url = 'https://github.com/LanaLob/randan/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['Sociology', 'SPSS', 'Statistical methods', 'CHAID'],
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'scikit-learn',
          'statsmodels',
          'factor_analyzer'
          'pydot',
          'graphviz'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)