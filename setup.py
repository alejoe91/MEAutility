# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

d = {}
exec(open("MEAutility/version.py").read(), None, d)
version = d['version']
long_description = open("README.md").read()

entry_points = None

install_requires = []

setup(
    name="MEAutility",
    version=version,
    author="Alessio Buccino",
    author_email="alessiop.buccino@gmail.com",
    description="Python package for multi-electrode array (MEA) handling and stimulation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/MEAutility",
    install_requires=[
        'numpy',
        'pyyaml',
        'matplotlib'
    ],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
)
