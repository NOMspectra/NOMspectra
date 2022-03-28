# masslib

small lib for working with FTCIR MS spectra

This lib supports
<ul>
    <li> Isotope Distribution generator </li>
    <li> Assigning brutto formulae to signal </li>
    <li> Working with spectra as with sets (intersection, union, etc) </li>
</ul>

<h2>Installation from source</h2>

You can install **masslib** with:

```sh
git clone https://github.com/nhslab/masslib/tree/another_branch

cd masslib

python setup.py install
```


<h2>Installation</h2>

For installing this lib it's only necessary to download project, and before using just add path for the lib to the path
lib_path = 'YOUR PATH TO THE LIB'

```python
lib_path = 'ENTER PATH TO THE LIB'
# for example:
# lib_path = '/home/dpreer/Polarwind/masslib'
# or it's possible to use relative path
# lib_path = 'Polarwind/masslib'

import sys
sys.path.insert(1, lib_path)
from mass import MassSpectrum
```

<h2>Usage Examples</h2>

Usage example for <b>isotope distribution generator</b>:
```python
from distribution_generation.mass_distribution import IsotopeDistribution

# brutto formulae that we want to use
brutto = {"Pd": 1, "Cl": 2}

# instance initialization
d = IsotopeDistribution(brutto)

# masses generations
d.generate_iterations(100000)

# plotting obtained distribution
d.draw()

# Graph can be saved by plt.savefig(filename, "png")
# For that import matplotlib.pyplot as plt is needed
```

Usage for <b>Arithmetic operations over assigned spectra</b>
```python
    
# imports
import os

import settings
from mass import MassSpectrum

#
masses = []
mapper = {"mw": "mass", "relativeAbundance": "I"}
for filename in sorted(os.listdir(settings.DATA_FOLDER)):
    masses.append(MassSpectrum().load(
        f"{settings.DATA_FOLDER}/{filename}",
        mapper=mapper,
        sep=',',
        ignore_columns=["peakNo", "errorPPM", "DBE", "class", "z"]
    ))
    
x, y, z = masses[:3]

union = (x + y + z)
print(len(union > 2))  # intersection of all three
print(len(union > 1))  # number of bruttos which is presented at least 2 out of 3 spectra

for i in [x, y, z]:
    print(union.calculate_jaccard_needham_score(i))

# deleting common for all 3 spectra
x -= (union > 2)
y -= (union > 2)
z -= (union > 2)
```

Usage for <b>Spectrum Assignment</b>
```python

import time
import pandas as pd
from mass import MassSpectrum

gen_brutto = pd.read_csv("../brutto_generator/C_H_O_N_S.csv", sep=";")

mapper = {"mw": "mass", "relativeAbundance": "I"}

T = time.time()

# in one expression we run:
# created instance of a mass spectrum
# load mass spectrum
# assign
# dropping unassigned bruttos
ms = MassSpectrum().load(
    "../data/a_1.csv",
    mapper,
    sep=',',
    ignore_columns=["peakNo", "errorPPM", "DBE", "class", "C", "H", "O", "N", "S", "z"]
).assign(gen_brutto, elems=list("CHONS")).drop_unassigned()

print(time.time() - T)
```

Usage to plot similarity matrix

```python
import os

import matplotlib.pyplot as plt

from mass import MassSpectrum, MassSpectrumList

# load data
masses = []
mapper = {"mw": "mass", "relativeAbundance": "I", "formula":"brutto"}
for filename in sorted(os.listdir("data"))[:5]:
    masses.append(MassSpectrum().load(
        f"data/{filename}",
        mapper=mapper,
        sep='\t',
        ignore_columns=["peakNo", "errorPPM", "DBE", "class", "z"]
    ).assignment_from_brutto())  # It's important, that similarity can be calculated only by formulae

# MassSpectrumList instance creating, 
collection = MassSpectrumList(masses, names=["1", "2", "3", "4", "5"])

# calculate and draw cosine similarity matrix
plt.figure(figsize=(12, 10))
collection.draw(collection.calculate_score(mode="common_correlation"), title="cosine")
plt.show()

# calculate and draw taminoto similarity matrix
plt.figure(figsize=(12, 10))
collection.draw(collection.calculate_score(mode="common_correlation"), title="tanimoto")
plt.show()

```

Usage for <b>Brutto table generation</b>


```python
import time
from brutto_generator import generate_brutto_formulas
T = time.time()
df = generate_brutto_formulas(
    min_n=(6, 6, 0, 0, 0),
    max_n=(40, 40, 5, 5, 5),
    elems=tuple("CHONS")
)
df.to_csv("test_C_H_O_N_S.csv", sep=";", index=False)
print(time.time() - T)

```

<h2>Building docs</h2>

```sh
pip install -r docs/requirements.txt

cd docs

make html
make epub
```