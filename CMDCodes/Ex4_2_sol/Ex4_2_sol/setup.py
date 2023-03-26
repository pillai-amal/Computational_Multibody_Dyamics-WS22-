from setuptools import setup, find_packages

name = "PyCMD"
version = "4.2"
author = "Giuseppe Capobianco"
author_email = (
    "giuseppe.capobianco@fau.de"
)
url = ""
description = "Computational multibody dynamics - exercise " + version
long_description = ""
license = "LICENSE"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=[
        "numpy>=1.21.3",
        "scipy>=1.6.1",
        "matplotlib>=3.4.3",
        # "black>=22.1.0",
        "tqdm>=4.62.3",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
)
