
[![PyPI](https://img.shields.io/pypi/v/natorgms)](https://pypi.org/project/natorgms/)
[![Documentation Status](https://readthedocs.org/projects/natorgms/badge/?version=latest)](https://natorgms.readthedocs.io/en/latest/?badge=latest)
[![Python package](https://github.com/natorgms/natorgms/actions/workflows/python-package.yml/badge.svg)](https://github.com/natorgms/natorgms/actions/workflows/python-package.yml)

# natorgms

![logo](https://github.com/natorgms/natorgms/raw/master/docs/_static/natorgms_logo.png)

natorgms is an open source Python package for processing high resolution mass spectra. Package designed for analysis dissolve organic matter, humic samples, petrolium and other difficult object which are characterized by thousands of signals in the spectrum and require special attention to analysis. The package is intended for processing initial mass lists, visualization spectra and determination of molecular descriptors such as aromasity index (AI), double bond equivavlen (DBE) and many others for further work.

Main operation:

- Assigning brutto formulas to signal by mass with desirable ranges of elements (include isotopes)
- Fine recallibrate spectrum by standart, asssigment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc)
- Plot spectrum and different kind of Scatter and density diagramm such as Van Krevelen diagram
- Calculate simmilarity metrics between spectra
- Calclulate moleculars descriptors (DBE, AI, NOSC, CRAM and other) for spectra

![figures](https://github.com/natorgms/natorgms/raw/master/docs/_static/gui_figures.jpg)

## Documentation

Documentation and examples of usage is placed [here](https://natorgms.readthedocs.io). Also you can explore jupyter [notebooks](https://github.com/natorgms/natorgms/tree/master/notebooks)

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

The GUI is pretty basic and not very flexible but it is support all basic operation.

## How to contribute or ask question

If you want to ask question or contribute to the development of natorgms, have a look at the [contribution guidelines](https://github.com/natorgms/natorgms/blob/master/CONTRIBUTING.md).

## License

Distributed under [license GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)