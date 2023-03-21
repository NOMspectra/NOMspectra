
[![PyPI](https://img.shields.io/pypi/v/nomspectra)](https://pypi.org/project/nomspectra/)
[![Documentation Status](https://readthedocs.org/projects/nomspectra/badge/?version=latest)](https://nomspectra.readthedocs.io/en/latest/?badge=latest)
[![Python package](https://github.com/nomspectra/nomspectra/actions/workflows/python-package.yml/badge.svg)](https://github.com/nomspectra/nomspectra/actions/workflows/python-package.yml)

# nomspectra

![logo](https://github.com/nomspectra/nomspectra/raw/master/docs/_static/nomspectra_logo.png)

nomspectra is an open-source Python package for processing high resolution mass spectra. The name is an acronym for Natural Organic Matter & Humic Substances Mass Spectrometry, so package designed for analysis natural organic matter (NOM) which are represent such substances as dissolve organic matter (DOM), humic substances, lignin, biochar and other objects of oxidative destruction of natural compounds which are characterized by thousands of signals in the mass spectrum and require special methods for analysis. The package implements a full-fledged workflow for processing, analysis and visualization of mass spectrum. Various algorithms for filtering spectra, recalibrating and assignment of elemental composition to ions are presented. The package implements methods for calculating different molecular descriptors and methods for data visualization. A graphical user interface (GUI) for package has been implemented, which makes this package convenient for a wide range of users.

Main operation:

- Assigning element composition to signal by mass with desirable ranges of elements (include isotopes)
- Fine recalibrate spectrum
- Working with spectra as with sets (intersection, union, etc.)
- Plot spectrum and different kind of scatter and density diagram such as Van Krevelen diagram
- Calculate similarity metrics between spectra
- Calculate molecular descriptors (DBE, AI, NOSC, CRAM and other) for spectra

![figures](https://github.com/nomspectra/nomspectra/raw/master/docs/_static/workflow.png)

## Documentation

Documentation and examples of usage is placed [here](https://nomspectra.readthedocs.io). Also, you can explore jupyter [notebooks](https://github.com/nomspectra/nomspectra/tree/master/notebooks)

## Install

Requirements:

- Python 3.8 or higher

Install nomspectra and dependences by pip:

```console
pip install nomspectra
```

## Graphical interface

After installation nomspectra you can run GUI by the command:

```console
python -m nomspectra
```

The GUI is pretty basic and not very flexible but it supports all basic operation.

## How to contribute or ask question

If you want to ask question or contribute to the development of nomspectra, have a look at the [contribution guidelines](https://github.com/nomspectra/nomspectra/blob/master/CONTRIBUTING.md).

## License

Distributed under [license GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)