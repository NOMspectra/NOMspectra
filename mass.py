import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple, Dict

import numpy as np
import pandas as pd

from utils import calculate_mass

logger = logging.getLogger(__name__)


class SpectrumIsNotAssigned(Exception):
    pass


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


class MassSpectrum(object):
    # should be columns: mass (!), I, calculated_mass, abs_error, rel_error

    def __init__(
            self,
            table: Optional[pd.DataFrame] = None,
            elems: Optional[list] = None
    ):
        self.elems = elems if elems else list("CHONS")
        self.features = ["mass", "calculated_mass", "I", "abs_error", "rel_error", "numbers"]

        if table is not None:
            self.table = table
            if "numbers" not in self.table:
                self.table["numbers"] = 1

        else:
            self.table = pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])

    def load(
            self,
            filename: Union[Path, str],
            mapper: Optional[Mapping[str, str]] = None,
            ignore_columns: Optional[Sequence[str]] = None,
            sep: str = ";"
    ) -> None:

        self.table = pd.read_csv(filename, sep=sep)
        if mapper:
            self.table = self.table.rename(columns=mapper)

        if ignore_columns:
            self.table = self.table.drop(columns=ignore_columns)

    def save(self, filename: Union[Path, str], sep: str = ";") -> None:
        self.table.to_csv(filename, sep=sep, index=False)

    def assign(
            self,
            generated_bruttos_table: pd.DataFrame,
            elems: Sequence[str],
            rel_error: float = 0.5
    ) -> "MassSpectrum":

        """Finding the nearest mass in generated_bruttos_table

        :param generated_bruttos_table: pandas DataFrame with column 'mass' and elements, should be sorted by 'mass'
        :param elems: Sequence of elements corresponding to generated_bruttos_table
        :param rel_error: error in ppm
        :return: MassSpectra object with assigned signals
        """

        overlap_columns = set(elems) & set(list(self.table))
        if overlap_columns:
            logger.warning(f"Following columns will be dropped: {overlap_columns}")
            table = self.table.drop(columns=elems)
        else:
            table = self.table.copy()

        masses = generated_bruttos_table["mass"]
        # masses -= 0.00054858  # electron mass

        elems = list(generated_bruttos_table.drop(columns=["mass"]))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = pd.DataFrame()
        for index, row in table.iterrows():
            mass = row["mass"]
            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1

            if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error:
                res = res.append({**dict(zip(elems, bruttos[idx])), "assign": True}, ignore_index=True)
            else:
                res = res.append({"assign": False}, ignore_index=True)

        return MassSpectrum(table.join(res))

    def __repr__(self):
        # repr only useful columns
        columns = [column for column in
                   ["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"] if column in self.table]

        return self.table[columns].__repr__()

    def __str__(self):
        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__str__()

    def calculate_error(self) -> "MassSpectrum":
        if "calculated_mass" not in self.table:
            table = self.calculate_mass()
        else:
            table = self.table.copy()

        table["abs_error"] = table["mass"] - table["calculated_mass"]
        table["rel_error"] = table["abs_error"] / table["mass"] * 1e6

        return MassSpectrum(table)

    def calculate_mass(self) -> "MassSpectrum":
        table = self.table.copy()
        table["calculated_mass"] = calculate_mass(self.table[self.elems].values, self.elems)
        return MassSpectrum(table)

    def get_brutto_list(self) -> Sequence[Tuple[float]]:
        return self.table[self.elems].values

    def get_brutto_dict(self) -> Mapping[Tuple, Dict[str, float]]:

        if len(self.table) == 0:
            return {}

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

    def __or__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        a = self.get_brutto_dict()
        b = other.get_brutto_dict()

        bruttos = set(a.keys()) | set(b.keys())

        # FIXME probably bad solution, hardcoded columns
        res = pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
        res = []
        for brutto in bruttos:
            if (brutto in a) and (brutto in b):
                # number is sum of a['numbers'] and b['numbers']
                c = a[brutto].copy()
                c["numbers"] += b[brutto]["numbers"]

                res.append(c)

            elif brutto in a:
                res.append(a[brutto])
            else:
                res.append(b[brutto])

        # FIXME probably bad solution, hardcoded columns
        res = pd.DataFrame(res) if len(res) > 0 else \
            pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
        bruttos = np.zeros((0, len(self.elems))) if len(bruttos) == 0 else list(bruttos)
        bruttos = pd.DataFrame(bruttos, columns=self.elems)

        res = pd.concat([res, bruttos], axis=1, sort=False).sort_values(by="mass")

        return MassSpectrum(res)

    def __xor__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        a = self.get_brutto_dict()
        b = other.get_brutto_dict()

        bruttos = set(a.keys()) ^ set(b.keys())

        res = []
        for brutto in bruttos:
            if brutto in a:
                res.append(a[brutto])
            else:
                res.append(b[brutto])

        # FIXME probably bad solution, hardcoded columns
        res = pd.DataFrame(res) if len(res) > 0 else \
            pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
        bruttos = np.zeros((0, len(self.elems))) if len(bruttos) == 0 else list(bruttos)
        bruttos = pd.DataFrame(bruttos, columns=self.elems)

        res = pd.concat([res, bruttos], axis=1, sort=False).sort_values(by="mass")

        return MassSpectrum(res)

    def __and__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        a = self.get_brutto_dict()
        b = other.get_brutto_dict()

        T = time.time()
        bruttos = set(a.keys()) & set(b.keys())
        print(T - time.time())

        res = []
        for brutto in bruttos:
            c = a[brutto].copy()
            c["numbers"] += b[brutto]["numbers"]
            res.append(c)

        # FIXME probably bad solution, hardcoded columns
        res = pd.DataFrame(res) if len(res) > 0 else \
            pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
        bruttos = np.zeros((0, len(self.elems))) if len(bruttos) == 0 else list(bruttos)
        bruttos = pd.DataFrame(bruttos, columns=self.elems)

        res = pd.concat([res, bruttos], axis=1, sort=False).sort_values(by="mass")

        return MassSpectrum(res)

    def __add__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        return self.__or__(other)

    def __sub__(self, other):
        a = self.get_brutto_dict()
        b = other.get_brutto_dict()

        bruttos = set(a.keys()) - set(b.keys())

        res = []
        for brutto in bruttos:
            res.append(a[brutto])

        # FIXME probably bad solution, hardcoded columns
        res = pd.DataFrame(res) if len(res) > 0 else \
            pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
        bruttos = np.zeros((0, len(self.elems))) if len(bruttos) == 0 else list(bruttos)
        bruttos = pd.DataFrame(bruttos, columns=self.elems)

        res = pd.concat([res, bruttos], axis=1, sort=False).sort_values(by="mass")

        return MassSpectrum(res)

    def __len__(self):
        return len(self.table)

    def __lt__(self, n: int) -> "MassSpectrum":
        return MassSpectrum(self.table[self.table["numbers"] < n])

    def __le__(self, n: int) -> "MassSpectrum":
        return MassSpectrum(self.table[self.table["numbers"] <= n])

    def __gt__(self, n: int) -> "MassSpectrum":
        return MassSpectrum(self.table[self.table["numbers"] > n])

    def __ge__(self, n: int) -> "MassSpectrum":
        return MassSpectrum(self.table[self.table["numbers"] >= n])

    def drop_unassigned(self) -> "MassSpectrum":
        if "assign" not in self.table:
            raise SpectrumIsNotAssigned()

        return MassSpectrum(self.table[self.table["assign"].astype(bool)])

    def reset_to_one(self) -> "MassSpectrum":
        table = self.table.copy()
        table["nubmers"] = 1

        return MassSpectrum(table)

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
