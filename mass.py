from itertools import *
from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple
import re

import numpy as np
import pandas as pd

import settings


class NoSuchChemicalElement(Exception):
    pass


monoisotopic_masses = {
    "C": 12.000000,
    "H": 1.08,
    "O": 15.999,
}


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


def generate_brutto_formulas(
    min_n: Sequence[int] = (-10, -30, -5, -5, -0),
    max_n: Sequence[int] = (10, 30, 5, 5, 0),
    elems: Union[str, Sequence[str]] = "CHONS"
):
    """
    Generates brutto formula by limit conditions and calculate masses
    :param min_n:
    :param max_n:
    :param elems: Iterable object of elements i.e ["CHO"] or ["Cu", "K", "C"]
    :return:
    """

    # generate brutto
    brutto = np.array(list(product(*[range(i, j + 1) for (i, j) in zip(min_n, max_n)])))

    # calculate masses
    masses = calculate_mass(brutto)

    # create pandas table for collect results
    df = pd.DataFrame()
    df["mass"] = masses

    for i, elem in enumerate(elems):
        df[elem] = brutto[:, i]

    # sorting table
    df = df.sort_values(by=["mass"])

    return df


class Brutto(object):
    def __init__(self, brutto: Union[str, Mapping[str, int]]) -> None:
        self.brutto = brutto

        if isinstance(brutto, str):
            if "_" in brutto:
                brutto = brutto.replace("_", "")

    def parse_brutto(self, brutto: str) -> Mapping[str, int]:
        """
        Ca3(PO4)2 -> {'Ca': 3, 'P': 4, 'O': 4}




        """

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def to_dict(self):
        pass

    def to_tuple(self):
        pass

    def elemets(self):
        pass


class MassSpectra(object):

    def __init__(self, table: Optional[pd.DataFrame] = None):
        if table:
            self.table = table

        else:
            self.table = pd.DataFrame()

    def load(self, filename: Union[Path, str], mapping: Mapping[str, str], sep=";"):
        self.table = pd.read_csv(filename, sep=sep)

        # should be columns: mass (!), I, brutto, calculated_mass, abs error, ppm error
        self.table = self.table.rename(mapping)

    def assign(self):
        pass

    def repr(self):
        return self.table

    def __str__(self):
        pass

    def calculate_error(self) -> None:
        pass

    def calculate_mass(self) -> None:
        pass

    def calculate_dbe(self) -> None:
        pass

    def calculate_ai(self) -> None:
        pass

    def get_list_brutto(self) -> Sequence[Tuple[]]:
        pass

    def __or__(self: "MassSpectr", other: "MassSpectr", mode="brutto") -> "MassSpectr":
        pass

    def __xor__(self: "MassSpectr", other: "MassSpectr") -> "MassSpectr":
        pass

    def __and__(self: "MassSpectr", other: "MassSpectr") -> "MassSpectr":
        pass

    def __add__(self: "MassSpectr", other: "MassSpectr") -> "MassSpectr":
        # by brutto

        a = self.get_list_brutto()
        b = other.get_list_brutto()

        # probably I can rewrite this piece

        a = set(a)
        b = set(b)


        elems = list("CHONS")
        table_1 = pd.DataFrame(list(a & b), columns=elems)
        table_1["numbers"] = 2

        table_2 = pd.DataFrame(list(a ^ b), columns=elems)
        table_2["numbers"] = 1

        return pd.concat([table_1, table_2])

    def __len__(self):
        return len(self.table)

    def calculate_jaccard_needham_score(self, other) -> float:
        return len(self & other) / len(self | other)

    def get_van_krevelen(self):
        pass

    def flat_van_krevelen(self):
        pass
