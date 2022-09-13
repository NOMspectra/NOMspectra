
[![PyPI](https://img.shields.io/pypi/v/nhsmass)](https://pypi.org/project/nhsmass/)
[![Documentation Status](https://readthedocs.org/projects/nhsmass/badge/?version=latest)](https://nhsmass.readthedocs.io/en/latest/?badge=latest)

# nhsmass

![logo](https://github.com/nhsmass/nhsmass/raw/master/docs/_static/nhsmass_logo.png)

nhsmass is an open source Python package for processing high resolution mass spectra, priviosly for analysis dissolve organic matter, humic samples, petrolium and other difficult object by FTICR or Orbitrap technique.

Main operation:

- Assigning brutto formulas to signal by mass with desirable ranges of elements (include isotopes)
- Recallibrate spectrum by etalon, asssigment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc)
- Plot spectrum and different kind of Van Krevelen diagramm
- Calculate simmilarity metrics and moleculars descriptors for spectra

![figures](https://github.com/nhsmass/nhsmass/raw/master/docs/_static/gui_figures.jpg)

## Documentation

Documentation and examples of usage is placed [here](https://nhsmass.readthedocs.io). Also you can explore jupyter [notebooks](https://github.com/nhsmass/nhsmass/tree/master/notebooks)

## Install

Requirements:

- Python 3.8

By pip:

```console
pip install nhsmass
```

## Graphical interface

After installation nhsmass you can run GUI by the command:

```console
python -m nhsmass
```

The GUI is pretty basic and not very flexible but it is support all basic operation.

## How to contribute

Feel free to use [issues](https://github.com/nhsmass/nhsmass/issues) to ask any question, make comments and suggestions.

## License

Distributed under [license GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)