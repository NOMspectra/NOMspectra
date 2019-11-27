from itertools import *
from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple, Dict
from typing import TypeVar

import re
from collections import Counter

import numpy as np
import pandas as pd

import settings

PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')


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
        br = Counter()
        pattern = "[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+"
        while "(" in brutto:
            for part in re.findall(pattern, brutto):
                pass

        for part in re.findall(pattern, brutto):
            if "(" in part:
                pass
            else:
                element = re.findall("[A-Z][a-z]", part)[0]
                number = re.findall("\d+", part)

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def to_dict(self) -> Dict[str, int]:
        pass

    def to_tuple(self):
        pass

    def elemets(self):
        pass


class MassSpectra(object):
    # should be columns: mass (!), I, calculated_mass, abs_error, rel_error

    def __init__(
            self,
            table: Optional[pd.DataFrame] = None,
            elems: Optional[list] = None
    ):
        self.elems = elems if elems else list("CHONS")
        self.features = ["mass", "calculated_mass", "I", "abs_error", "rel_error", "numbers"]

        if table:
            self.table = table
            if "numbers" not in self.table:
                self.table["numbers"] = 1

        else:
            self.table = pd.DataFrame()

    def load(
            self,
            filename: Union[Path, str],
            mapper: Optional[Mapping[str, str]] = None,
            sep: str = ";"
    ) -> None:

        self.table = pd.read_csv(filename, sep=sep)
        if mapper:
            self.table = self.table.rename(mapper)

    def save(self, filename: Union[Path, str], sep: str = ";"):

        self.table.to_csv(filename, sep=sep)

    def assign(self, generated_bruttos_table: pd.DataFrame, elems: Sequence[str], rel_error: float = 0.5) -> "MassSpectra":
        """Finding the nearest mass in generated_bruttos_table

        :param generated_bruttos_table: pandas DataFrame with column 'mass' and elements, should be sorted by 'mass'
        :param elems: Sequence of elements corresponding to generated_bruttos_table
        :param rel_error: error in ppm
        :return: MassSpectra object with assigned signals
        """
        table = self.table.copy()
        masses = generated_bruttos_table["mass"]

        elems = list(generated_bruttos_table.drop("mass"))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = pd.DataFrame()
        for row in table.iterrows():
            mass = row["mass"]
            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1

            if np.fabs(masses[idx] - mass) / mass <= rel_error:
                res.append({**dict(zip(elems, bruttos[idx])), "assign": True}, ignore_index=True)
            else:
                res.append({"assign": False}, ignore_index=True)

        return MassSpectra(pd.concat([table, res], axis=1, ignore_index=True))

    def __repr__(self):
        # repr only useful columns
        columns = ["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"]
        return self.table[columns].__repr__()

    def __str__(self):
        return self.table[self.features].__str__()

    def calculate_error(self) -> "MassSpectra":
        if "calculated_mass" not in self.table:
            table = self.calculate_mass()
        else:
            table = self.table.copy()

        table["abs_error"] = table["mass"] - table["calculated_mass"]
        table["rel_error"] = table["abs_error"] / table["mass"]

        return MassSpectra(table)

    def calculate_mass(self) -> "MassSpectra":
        table = self.table.copy()
        table["calculated_mass"] = calculate_mass(self.table[self.elems].values, self.elems)
        return MassSpectra(table)

    def get_list_brutto(self) -> Sequence[Tuple[float]]:
        return self.table[self.elems].values

    def get_dict_brutto(self) -> Mapping[Tuple, Dict[str, float]]:

        res = {}
        bruttos = self.table[self.elems].values.tolist()
        bruttos = [tuple(brutto) for brutto in bruttos]

        for [mass, calculated_mass, I, abs_error, rel_error, numbers], brutto \
                in zip(self.table[self.features].values, bruttos):

            res[brutto] = {
                "mass": mass,
                "calculated_mass": calculated_mass,
                "I": I,
                "abs_error": abs_error,
                "rel_error": rel_error,
                "numbers": numbers
            }

        return res

    def __or__(self: "MassSpectra", other: "MassSpectra") -> "MassSpectra":
        a = self.get_dict_brutto()
        b = self.get_dict_brutto()

        bruttos = set(a.keys()) | set(b.keys())

        res = pd.DataFrame()
        for brutto in bruttos:
            if brutto in a and brutto in b:
                # number is sum of a['numbers'] and b['numbers']
                c = a[brutto].copy()
                c["numbers"] += b[brutto]["numbers"]

                res.append(c, ignore_index=True)

            elif brutto in a:
                res.append(a[brutto], ignore_index=True)
            else:
                res.append(b[brutto], ignore_index=True)

        bruttos = pd.DataFrame(np.array(bruttos), columns=self.elems)
        res = pd.concat([res, bruttos], axis=1, sort=False).sort(by="mass")

        return MassSpectra(res)

    def __xor__(self: "MassSpectra", other: "MassSpectra") -> "MassSpectra":
        a = self.get_dict_brutto()
        b = self.get_dict_brutto()

        bruttos = set(a.keys()) ^ set(b.keys())

        res = pd.DataFrame()
        for brutto in bruttos:
            if brutto in a:
                res.append(a[brutto], ignore_index=True)
            else:
                res.append(b[brutto], ignore_index=True)

        bruttos = pd.DataFrame(np.array(bruttos), columns=self.elems)
        res = pd.concat([res, bruttos], axis=1, sort=False).sort(by="mass")

        return MassSpectra(res)

    def __and__(self: "MassSpectra", other: "MassSpectra") -> "MassSpectra":
        a = self.get_dict_brutto()
        b = self.get_dict_brutto()

        bruttos = set(a.keys()) & set(b.keys())

        res = pd.DataFrame()
        for brutto in bruttos:
            c = a[brutto].copy()
            c["numbers"] += b[brutto]["numbers"]
            res.append(c, ignore_index=True)

        bruttos = pd.DataFrame(np.array(brutto), columns=self.elems)
        res = pd.concat([res, bruttos], axis=1, sort=False).sort(by="mass")

        return MassSpectra(res)

    def __add__(self: "MassSpectra", other: "MassSpectra") -> "MassSpectra":
        return self.__or__(other)

    def __sub__(self, other):
        a = self.get_dict_brutto()
        b = self.get_dict_brutto()

        bruttos = set(a.keys()) - set(b.keys())

        res = pd.DataFrame()
        for brutto in bruttos:
            res.append(a[brutto], ignore_index=True)

        bruttos = pd.DataFrame(np.array(brutto), columns=self.elems)
        res = pd.concat([res, bruttos], axis=1, sort=False).sort(by="mass")

        return MassSpectra(res)

    def __len__(self):
        return len(self.table)

    def __lt__(self, n: int) -> "MassSpectra":
        return MassSpectra(self.table[self.table["numbers"] < n])

    def __le__(self, n: int) -> "MassSpectra":
        return MassSpectra(self.table[self.table["numbers"] <= n])

    def __gt__(self, n: int) -> "MassSpectra":
        return MassSpectra(self.table[self.table["numbers"] > n])

    def __ge__(self, n: int) -> "MassSpectra":
        return MassSpectra(self.table[self.table["numbers"] >= n])

    def calculate_jaccard_needham_score(self, other) -> float:
        return len(self & other) / len(self | other)

    def get_van_krevelen(self):
        pass

    def flat_van_krevelen(self):
        pass

    def calculate_dbe(self) -> None:
        pass

    def calculate_ai(self) -> None:
        pass
