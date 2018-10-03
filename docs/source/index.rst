.. MEAutility documentation master file, created by
   sphinx-quickstart on Wed Oct  3 16:27:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MEAutility's documentation!
======================================

Python package for multi-electrode array (MEA) handling and stimulation.

Installation
============

To install run:

.. code:: ipython3

    pip install MEAutility

If you want to install from sources and be updated with the latest development you can install with:

.. code:: ipython3

    git clone https://github.com/alejoe91/MEAutility
    cd MEAutility
    python setup.py install (or develop)
    


The package can then imported in Python with:

.. code:: ipython3

    import MEAutility as MEA

Requirements
============

- numpy
- pyyaml
- matplotlib


Contents
========

The following sections will guide you through definitions and handling of MEA models, as well as electrical stimulation
and plotting functions.

.. toctree::
   :maxdepth: 2

   mea_definitions
   mea_handling
   mea_stimulation
   mea_plotting


Contact
=======

If you have questions or comments, contact Alessio Buccino: alessiob@ifi.uio.no