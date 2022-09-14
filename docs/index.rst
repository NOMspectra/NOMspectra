.. natorgms documentation master file, created by
   sphinx-quickstart on Thu Aug 11 18:33:15 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

natorgms documentation
====================================

.. image:: _static/natorgms_logo.png
  :alt: natorgms_logo

natorgms is an open-source Python package for processing high resolution mass spectra. Package designed for analysis dissolve organic matter, humic samples, petroleum and other difficult object which are characterized by thousands of signals in the spectrum and require special attention to analysis. The package is intended for processing initial mass lists, visualization spectra and determination of molecular descriptors such as aromaticity index (AI), double bond equivalent (DBE) and many others for further work.

Main operation:

- Assigning brutto formulas to signal by mass with desirable ranges of elements (include isotopes)
- Fine recalibrate spectrum by standard, assignment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc.)
- Plot spectrum and different kind of Scatter and density diagram such as Van Krevelen diagram
- Calculate similarity metrics between spectra
- Calculate molecular descriptors (DBE, AI, NOSC, CRAM and other) for spectra

.. image:: _static/gui_figures.jpg
  :alt: gui_figures

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   Install <install.md>
   Examples <examples.rst>
   API <api/natorgms.rst>
   GUI Tutorial <gui_tutorial.md>
