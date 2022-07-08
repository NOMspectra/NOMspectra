# masslib

version 0.1.0

small lib for working with high resolution mass spectrums

This lib supports

- Plot spectrum and different kind od Van Krevelen diagramm
- Assigning brutto formulae to signal by mass and TMDS
- Recallibrate spectrum by etalon, asssigment error or dif-mass map
- Working with spectra as with sets (intersection, union, etc)
- Calculate and plot simmilarity map of spectrums

## Installation

download masslib by clone this repository

then for install localy by pip:

```console
pip install .
```

## Examples of usage

There are few jupyter notebook that explain how to use masslib. It is lockated in folder "ipynb examples"

## Graphical interface

There is small script for graphical interface lockated in folder "gui/masslib_gui.py"

It is support all main operation under spectrum

It's additional require pyQt and matplotlib-matplotlib-venn so you need to install it by pip

```console
pip install pyqt5 matplotlib-venn
```