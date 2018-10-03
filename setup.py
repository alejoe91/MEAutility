# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import MEAutility

long_description = open("README.md").read()

entry_points = None

install_requires = []

setup(
    name="MEAutility",
    version=MEAutility.__version__,
    author="Alessio Buccino",
    author_email="alessiob@ifi.uio.no",
    description="Python package for multi-electrode array (MEA) handling and stimulation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alejoe91/MEAutility",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    entry_points=entry_points,
    include_package_data=True,
)
