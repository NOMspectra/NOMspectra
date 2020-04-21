import logging
from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple, Dict, SupportsFloat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid.inset_locator import inset_axes as inset_axes_func

from brutto import Brutto
from brutto_generator import brutto_gen_dummy
from utils import calculate_mass
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SpectrumIsNotAssigned(Exception):
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
            names: Optional[Sequence[str]] = None,
            sep: str = ";"
    ) -> "MassSpectrum":
        self.table = pd.read_csv(filename, sep=sep, names=names)
        if mapper:
            self.table = self.table.rename(columns=mapper)

        if ignore_columns:
            self.table = self.table.drop(columns=ignore_columns)

        if "numbers" not in self.table:
            self.table["numbers"] = 1

        return self

    def save(self, filename: Union[Path, str], sep: str = ";") -> None:
        """Saves to csv MassSpectrum"""
        self.table.to_csv(filename, sep=sep, index=False)

    def assign(
            self,
            generated_bruttos_table: pd.DataFrame,
            elems: Sequence[str],
            rel_error: float = 0.5,
            sign='-'
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

        if sign == '-':
            masses -= 0.00054858  # electron mass
        if sign == '+':
            masses += 0.00054858  # electron mass

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

    def assign_dummy(
            self,
            elems: str = 'CHONS',
            rel_error: float = 0.5,
            sign='-',
            round_search: int = 5
    ) -> "MassSpectrum":

        """Finding the nearest mass in generated_bruttos_table

        :param elems: Sequence of elements corresponding to generated_bruttos_table
        :param rel_error: error in ppm
        :param sign: mode of spectrum '-' means subtracting the mass of the electron and hydrogen, '+' means adding electron mass
        :param round search: error of mass calculate 10^(-round_search)
        :return: MassSpectra object with assigned signals
        """

        overlap_columns = set(elems) & set(list(self.table))
        if overlap_columns:
            logger.warning(f"Following columns will be dropped: {overlap_columns}")
            table = self.table.drop(columns=elems)
        else:
            table = self.table.copy()

        generated_bruttos_dict = brutto_gen_dummy(elems, round_search)
        elem_order = {'C':0, 'H':1, 'O':2, 'N':3, 'S':4}

        for elem in elems:
            table[elem] = np.NaN
        table['assign'] = 0
        
        if sign == '-':
            mass_shift = - 0.00054858 - 1.007825 # electron and hydrogen mass

        if sign == '+':
            mass_shift = 0.00054858  # electron mass

        step = pow(10, -round_search)
        for i in range(table.shape[0]):
            mass = table.loc[i, 'mass'] + mass_shift 
            mass_error = mass * rel_error * 0.000001
            
            #formula find in brutto generator, search started with smallest error 
            mass_dif = 0
            chons = []
            while mass_dif < mass_error:
                if round(mass + mass_dif, round_search) in generated_bruttos_dict:
                    chons = generated_bruttos_dict[round(mass + mass_dif, round_search)]
                    break
                elif round(mass - mass_dif, round_search) in generated_bruttos_dict:
                    chons = generated_bruttos_dict[round(mass - mass_dif, round_search)]
                    break
                else:
                    mass_dif += step
        
            if len(chons) > 0:
                for elem in 'CHONS':
                    table.at[i, elem] = chons[elem_order[elem]]
                table.at[i, 'assign'] = 1    
                    
        return MassSpectrum(table)

    def assignment_from_brutto(self) -> 'MassSpectrum':
        if "brutto" not in self.table:
            raise Exception("There is no brutto in MassSpectra")

        # before new assignment it's necessary to drop old assignment
        table = self.table.drop(columns=self.elems)

        elems = set.union(*[set(list(x)) for x in self.table.brutto.apply(lambda x: x.replace("_", "")).apply(
            lambda x: Brutto(x).get_elements()).tolist()])

        for element in elems:
            table[element] = table.brutto.apply(lambda x: Brutto(x.replace("_", ""))[element])

        return MassSpectrum(table, elems=elems)

    def compile_brutto(self) -> 'MassSpectrum':
        def compile_one(a: Sequence[Union[int, float]], elems: Sequence[str]) -> str:
            s = ''
            for c, e in zip(a, elems):
                if not np.isnan(c):
                    s += e + ('' if c == 1 else str(int(c)))
                else:
                    # if coefficients is Not a Number (is nan)
                    # formula is unknown
                    return ''
            return s

        table = self.table.copy()

        # iterations by rows, so axis=1
        table["brutto"] = self.table[self.elems].apply(lambda x: compile_one(x.values, self.elems), axis=1)

        return MassSpectrum(table)

    def copy(self) -> 'MassSpectrum':
        return MassSpectrum(self.table)

    def __repr__(self):
        # repr only useful columns
        columns = [column for column in
                   ["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"] if column in self.table]

        return self.table[columns].__repr__()

    def __str__(self) -> str:
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

    def get_brutto_dict(self, elems: Optional[Sequence[str]] = None) -> Mapping[Tuple, Dict[str, float]]:

        if len(self.table) == 0:
            return {}

        res = {}

        # TODO very careful
        if elems:
            for elem in elems:
                # careful change the object
                if elem not in self.table:
                    self.table[elem] = 0

            bruttos = self.table[elems].values.tolist()
        else:
            bruttos = self.table[self.elems].values.tolist()

        bruttos = [tuple(brutto) for brutto in bruttos]

        columns = list(self.table.drop(columns=self.elems))
        for values, brutto in zip(self.table[columns].values, bruttos):
            res[brutto] = dict(zip(columns, values.tolist()))

        return res

    def __or__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        a = self.get_brutto_dict()
        b = other.get_brutto_dict()

        bruttos = set(a.keys()) | set(b.keys())

        # FIXME probably bad solution, hardcoded columns
        # res = pd.DataFrame(columns=["I", "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])
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

        bruttos = set(a.keys()) & set(b.keys())

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

    def __len__(self) -> int:
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
        table["numbers"] = 1

        return MassSpectrum(table)

    def calculate_jaccard_needham_score(self, other) -> float:
        return len(self & other) / len(self | other)

    def calculate_ai(self) -> 'MassSpectrum':
        table = self.calculate_cai().calculate_dbe().table
        table["AI"] = table["DBE"] / table["CAI"]

        return MassSpectrum(table)

    def calculate_cai(self) -> 'MassSpectrum':
        table = self.table.copy()

        # very careful
        # anyway it's necessary to have at least column with C?
        for element in "CONSP":
            if element not in table:
                table[element] = 0

        table['CAI'] = table["C"] - table["O"] - table["N"] - table["S"] - table["P"]

        return MassSpectrum(table)

    def calculate_dbe(self) -> 'MassSpectrum':
        table = self.table.copy()
        table['DBE'] = 1.0 + table["C"] - table["O"] - table["S"] - 0.5 * table["H"]

        return MassSpectrum(table)

    def __getitem__(self, item: Union[str, Sequence[str]]) -> pd.Series:
        return self.table[item]

    def normalize(self) -> 'MassSpectrum':
        """This function return intensity normalized MassSpectrum instance"""

        table = self.table.copy()
        table["I"] /= table["I"].max()
        return MassSpectrum(table["I"])

    def head(self) -> pd.DataFrame:
        return self.table.head()

    def tail(self) -> pd.DataFrame:
        return self.table.tail()

    def draw(self,
             xlim: Tuple[Optional[float], Optional[float]] = (None, None),
             ylim: Tuple[Optional[float], Optional[float]] = (None, None),
             color: str = 'black'
    ) -> None:

        df = self.table.sort_values(by="mass")

        mass = df.mass.values
        if xlim[0] is None:
            xlim = (mass.min(), xlim[1])
        if xlim[1] is None:
            xlim = (xlim[0], mass.max())

        intensity = df.I.values
        # filter first intensity and only after mass (because we will lose the information)
        intensity = intensity[(xlim[0] <= mass) & (mass <= xlim[1])]
        mass = mass[(xlim[0] <= mass) & (mass <= xlim[1])]

        # bas solution, probably it's needed to rewrite this piece
        M = np.zeros((len(mass), 3))
        M[:, 0] = mass
        M[:, 1] = mass
        M[:, 2] = mass
        M = M.reshape(-1)

        I = np.zeros((len(intensity), 3))
        I[:, 1] = intensity
        I = I.reshape(-1)

        plt.plot(M, I, color=color)
        plt.plot([xlim[0], xlim[1]], [0, 0], color=color)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("m/z, Da")
        plt.ylabel("Intensity")

        return

    def intergrate(self, normalize=False) -> Tuple[np.ndarray, np.ndarray]:
        tmp = self.table[["mass"]]
        tmp = tmp.append({"mass": 0}, ignore_index=True).sort_values("mass")
        tmp["index"] = pd.RangeIndex(start=0, stop=len(tmp), step=1)
        if normalize:
            tmp['index'] /= len(tmp)

        return tmp["mass"].values, tmp["index"].values


class CanNotCreateVanKrevelen(Exception):
    pass


class VanKrevelen(object):
    def __init__(self, table: Optional[pd.DataFrame] = None, name: Optional[str] = None):
        self.name = name

        if not (("C" in table and "H" in table and "O" in table) or ("O/C" in table or "H/C" in table)):
            raise CanNotCreateVanKrevelen()

        self.table = table
        if "O/C" not in self.table:
            self.table["O/C"] = self.table["O"] / self.table["C"]

        if "H/C" not in self.table:
            self.table["H/C"] = self.table["H"] / self.table["C"]

    def draw_density(self, cmap="Blues", ax=None, shade=True):
        sns.kdeplot(self.table["O/C"], self.table["H/C"], ax=ax, cmap=cmap, shade=shade)

    def draw_density_with_marginals(self, color=None, ax=None):
        sns.jointplot(x="O/C", y="H/C", data=self.table, kind="kde", color=color, ax=ax)

    def draw_scatter_with_marginals(self):
        sns.jointplot(x="O/C", y="H/C", data=self.table, kind="scatter")

    def draw_scatter(self, ax=None, s=10, alpha=0.1):
        if ax:
            ax.scatter(self.table["O/C"], self.table["H/C"], s=s, alpha=alpha)
            ax.set_xlabel("O/C")
            ax.set_ylabel("H/C")
        else:
            plt.scatter(self.table["O/C"], self.table["H/C"], s=10)
            plt.xlabel("O/C")
            plt.ylabel("H/C")

    def boxed_van_krevelen(self, r=5, c=4) -> Sequence[Sequence]:
        # (array([0.2, 0.6, 1. , 1.4, 1.8, 2.2]), array([0.  , 0.25, 0.5 , 0.75, 1.  ]))

        df = self.table
        x = np.linspace(0.2, 2.2, r + 1)  # 0.4
        y = np.linspace(0, 1, c + 1)  # 0.25

        vc = []
        for i in range(r):
            vc.append([])
            for j in range(c):
                vc[-1].append(
                    df[
                        (df["H/C"] > x[i]) &
                        (df["H/C"] <= x[i] + 2.0 / r) &
                        (df["O/C"] > y[j]) &
                        (df["O/C"] <= y[j] + 1.0 / c)
                    ])

        return vc

    def density_boxed_van_crevelen(self, r=5, c=4):
        vc = self.boxed_van_krevelen(r=r, c=c)

        res = np.zeros((r, c))
        for i in range(len(vc)):
            for j in range(len(vc[0])):
                res[i][j] = len(vc[i][j])

        res = np.array(res)
        res /= np.sum(res)

        return res


def calculate_ppm(x: float, y: float) -> float:
    return np.fabs((x - y) / y * 1e6)


class MassSpectrumList(object):
    def __init__(self, spectra: Sequence[MassSpectrum], names: Optional[Sequence[str]] = None):
        self.spectra = spectra
        if names:
            self.names = names
        else:
            self.names = list(range(len(spectra)))

        self.elems = self.find_elems()
        self.pivot = self.calculate_pivot()

    def calculate_pivot_without_brutto(self, delta_ppm: float = 1) -> pd.DataFrame:
        masses = []
        for spectrum in self.spectra:
            masses.append(spectrum["mass"].values)

        masses = np.concatenate(masses)
        masses = np.sort(masses)
        clusters = []
        print("SORTED")
        for mass in masses:
            if len(clusters) == 0:
                clusters.append([mass])

            else:
                if calculate_ppm(clusters[-1][-1], mass) < delta_ppm:
                    clusters[-1].append(mass)

                else:
                    clusters.append([mass])

        print(f"Median Calculating, len(clusters) = {len(clusters)}")
        median_masses = []
        for cluster in clusters:
            median_masses.append(np.median(cluster))

        table = []
        print("Search begins")
        for mass in tqdm(median_masses):
            table.append([])
            for spectrum in self.spectra:
                masses = spectrum["mass"].values
                idx = np.searchsorted(masses, mass, side='left')
                if idx > 0 and (idx == len(masses) or (np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx]))):
                    idx -= 1

                if calculate_ppm(masses[idx], float(mass)) < delta_ppm:
                    table[-1].append(spectrum["I"].values[idx])
                else:
                    table[-1].append(0)

        df = pd.DataFrame(table, columns=self.names)
        df['mass'] = median_masses
        return df[['mass'] + self.names]

    def load_from_table(
            self,
            pivot: pd.DataFrame,
            names: Sequence[str],
            elems: Optional[Sequence[str]] = None
    ) -> None:
        self.names = names
        self.pivot = pivot
        if None:
            self.elems = elems

    def find_elems(self):
        elems = set([])
        for spectra in self.spectra:
            elems.update(set(spectra.elems))

        return list(elems)

    def calculate_pivot(self) -> pd.DataFrame:
        spectra = []

        for spectrum in self.spectra:
            spectra.append(spectrum.get_brutto_dict(elems=self.elems))

        bruttos = set()
        for spectrum in spectra:
            bruttos.update(set(spectrum.keys()))

        pivot = []
        for brutto in bruttos:
            vector = []
            for spectrum in spectra:
                vector.append(spectrum[brutto]["I"] if brutto in spectrum else 0)

            pivot.append(vector)

        pivot = pd.DataFrame(pivot, columns=self.names)
        for i, elem in enumerate(self.elems):
            pivot[elem] = [brutto[i] for brutto in bruttos]

        return pivot

    def calculate_similarity(self, mode: str = "taminoto") -> np.ndarray:
        """Calculate similarity matrix for all spectra in MassSpectrumList

        :param mode: one of the similarity functions.
        Mode can be: "tanimoto", "jaccard", "correlation", "common_correlation"

        :return: similarity matrix, 2d np.ndarray with size [len(names), len(names)]"""

        def jaccard(a, b):
            a = a.astype(bool)
            b = b.astype(bool)

            return (a & b).sum() / (a | b).sum()

        def tanimoto(a, b):
            return (a * b).sum() / ((a * a).sum() + (b * b).sum() - (a * b).sum())

        def common_correlation(a, b):
            A = a[a.astype(bool) == b.astype(bool)]
            B = b[a.astype(bool) == b.astype(bool)]

            return np.corrcoef(A, B)[0, 1]

        def cosine(a, b):
            return (a*b).sum() / ((a*a).sum() * (b*b).sum())**0.5

        def correlation(a, b):
            return np.corrcoef(a, b)[0, 1]

        if mode == "jaccard":
            similarity_func = jaccard
        elif mode == "tanimoto":
            similarity_func = tanimoto
        elif mode == "correlation":
            similarity_func = correlation
        elif mode == "common_correlation":
            similarity_func = common_correlation
        elif mode == 'cosine':
            similarity_func = cosine
        else:
            raise Exception(f"There is no such mode: {mode}")

        df = self.pivot
        values = []
        for i in self.names:
            values.append([])
            for j in self.names:
                values[-1].append(similarity_func(df[i], df[j]))

        return np.array(values)

    def calculate_dissimilarity(self, mode: str = "tanimoto"):
        return 1 - self.calculate_dissimilarity(mode=mode)

    def draw(
            self,
            values: np.ndarray,
            title: str = ""
    ) -> None:
        """Draw similarity matrix by using seaborn

        :param values: similarity matix
        :param title: title for plot
        :return: None"""

        sns.heatmap(np.array(values), vmin=0, vmax=1, annot=True)

        plt.xticks(np.arange(len(self.names)) + .5, self.names, rotation="vertical")
        plt.yticks(np.arange(len(self.names)) + .5, self.names, rotation="horizontal")
        plt.title(title)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.xlim(b, t)

    def draw_mass_spectrum(self, fig=None):
        """
        TODO NEED TO BE TESTED

        :param fig:
        :return:
        """
        spectra = self.spectra
        if fig is None:
            fig = plt.figure(figsize=(15, 8))
        for i, spectrum in enumerate(spectra):
            ax = fig.add_subplot(len(spectra) // 4 + int((len(spectra) % 4) > 0), 4, i + 1)

            # plt.title(spectrum) - I'm a little bit unsure here
            spectrum.normalize().draw(xlim=(200, 1000))
            inset_axes = inset_axes_func(ax,
                                         width="40%",  # width = 30% of parent_bbox
                                         height="40%",  # height : 1 inch
                                         )
            spectrum.draw(xlim=(385, 385.225))
            plt.xticks([385., 385.1, 385.2])

        plt.tight_layout()


if __name__ == '__main__':
    ms = MassSpectrum().load('tests/test.csv').drop_unassigned()

    vk = VanKrevelen(ms.table, name="Test VK")
    # vk.draw_density()

    print(vk.density_boxed_van_crevelen())
    plt.show()
