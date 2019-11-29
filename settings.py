import os.path
from pathlib import Path

# Get the absolute path of the settings.py file's directory
PWD = os.path.dirname(os.path.realpath(__file__ ))

MONOISOTOPIC_MASSES_PATH = Path(PWD) / "masses" / "monoisotopic_masses.csv"
ISOTOPE_ABUNDANCE_PATH = Path(PWD) / "masses" / "isotope_abundance.csv"
AVERAGE_MASSES_PATH = Path(PWD) / "masses" / "average_masses.csv"
DATA_FOLDER = Path(PWD) / "data"
