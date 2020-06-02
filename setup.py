import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="randan-Lana_Lob", # Replace with your own username
    version="0.0.1",
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
    python_requires='>=3.6',
)
