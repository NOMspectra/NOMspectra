.. nhsmass documentation master file, created by
   sphinx-quickstart on Thu Aug 11 18:33:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nhsmass's documentation!
===================================

.. image:: _static/nhsmass_logo.png
  :width: 100
  :alt: nhsmass_logo

nhsmass is an open-access Python package for processing high resolution mass spectra, priviosly for analysis dissolve organic matter, humic samples and other difficult object by FTICR or Orbitrap technique. 

Main operation:

- Assigning brutto formulas to signal by mass with desirable ranges of elements (include isotopes)
- Recallibrate spectrum by etalon, asssigment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc)
- Plot spectrum and different kind of Van Krevelen diagramm
- Calculate simmilarity metrics and moleculars descriptors for spectra
- and many other usefull calculation

.. toctree::
   :maxdepth: 3
   :caption: API:

   modules

Installation
============

Requirements:

- Python 3.8

Install nhsmass by pip

.. code-block:: console

  pip install nhsmass

Examples of usage
=================

There are few jupyter notebook that explain how to use nhsmass. It is lockated in folder "notebooks" in Github repo:
https://github.com/nhsmass/nhsmass/tree/master/notebooks

Graphical interface
===================

Simple GUI is lockated here

https://github.com/nhsmass/nhsmassPQ

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
