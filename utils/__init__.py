import pandas as pd
from typing import Sequence, Union
import numpy as np
import settings


class NoSuchChemicalElement(Exception):
    pass


def calculate_mass(
    brutto_formulas: Sequence[Sequence],
    elems: Union[str, Sequence[str]] = "CHONS"
) -> np.ndarray:
    """
    Calculate monoisotopic masses for sequence of brutto formulae coefficients tuple

    :param brutto_formulas: 2d array of size (number of brutto formulae, len(elem))
    :param elems: elements that corresponds to columns
    :return: sequence of calculated masses
    """
    masses = pd.read_csv(settings.MONOISOTOPIC_MASSES_PATH, sep=";")
    elem_masses = []
    for elem in elems:
        try:
            mass = masses[masses.element == elem]["mass"].values[0]
        except Exception as e:
            raise NoSuchChemicalElement(f"There is not element: {elem},\nerror: {e}")
        elem_masses.append(mass)

    elem_masses = np.array(elem_masses)
    return np.sum(np.array(brutto_formulas) * elem_masses, axis=1)


if __name__ == '__main__':
    calculate_mass([()])
