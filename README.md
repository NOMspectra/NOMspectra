
[![PyPI](https://img.shields.io/pypi/v/natorgms)](https://pypi.org/project/natorgms/)
[![Documentation Status](https://readthedocs.org/projects/natorgms/badge/?version=latest)](https://natorgms.readthedocs.io/en/latest/?badge=latest)
[![Python package](https://github.com/natorgms/natorgms/actions/workflows/python-package.yml/badge.svg)](https://github.com/natorgms/natorgms/actions/workflows/python-package.yml)

# natorgms

![logo](https://github.com/natorgms/natorgms/raw/master/docs/_static/natorgms_logo.png)

natorgms is an open-source Python package for processing high resolution mass spectra. The name is an acronym for Natural Organic matter Mass Spectrometry, so package designed for analysis natural organic matter (NOM) which are represent such substances as dissolve organic matter (DOM), humic substances, lignin, biochar and other objects of oxidative destruction of natural compounds which are characterized by thousands of signals in the spectrum and require special methods for analysis. The package implements a full-fledged workflow for processing, analysis and visualization of mass spectrum. Various algorithms for filtering spectra, recalibrating and assignment of elemental composition to ions are presented. The package implements methods for calculating different molecular descriptors and methods for data visualization. A graphical user interface (GUI) for package has been implemented, which makes this package convenient for a wide range of users.

Main operation:

- Assigning brutto formulas to signal by mass with desirable ranges of elements (include isotopes)
- Fine recalibrate spectrum by standard, assignment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc.)
- Plot spectrum and different kind of Scatter and density diagram such as Van Krevelen diagram
- Calculate similarity metrics between spectra
- Calculate molecular descriptors (DBE, AI, NOSC, CRAM and other) for spectra

![figures](https://github.com/natorgms/natorgms/raw/master/docs/_static/workflow.png)

## Documentation

Documentation and examples of usage is placed [here](https://natorgms.readthedocs.io). Also, you can explore jupyter [notebooks](https://github.com/natorgms/natorgms/tree/master/notebooks)

## Install

Requirements:

- Python 3.8 or higher

Install natorgms and dependences by pip:

```console
pip install natorgms
```

## Graphical interface

After installation natorgms you can run GUI by the command:

```console
python -m natorgms
```

The GUI is pretty basic and not very flexible but it supports all basic operation.

## How to contribute or ask question

If you want to ask question or contribute to the development of natorgms, have a look at the [contribution guidelines](https://github.com/natorgms/natorgms/blob/master/CONTRIBUTING.md).

## License

Distributed under [license GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)