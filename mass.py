import logging
from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import copy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.stats as st
from mpl_toolkits.axes_grid.inset_locator import inset_axes as inset_axes_func

from brutto import Brutto
from brutto_generator import brutto_gen
from tqdm import tqdm

logger = logging.getLogger(__name__)

masses_path='masses/element_table.csv'

class SpectrumIsNotAssigned(Exception):
    pass


class MassSpectrum(object):
    # should be columns: mass (!), I, calculated_mass, abs_error, rel_error

    def __init__(
            self,
            table: Optional[pd.DataFrame] = None,
            elems: Optional[list] = None,
    ):
        self.features = ["mass", "calculated_mass", 'intensity', "abs_error", "rel_error", "numbers"]

        if table is not None:
            self.table = table
            if "numbers" not in self.table:
                self.table["numbers"] = 1
        else:
            self.table = pd.DataFrame(columns=['intensity', "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])

        if elems:
            self.elems = elems
        else:
            self.elems = self.find_elems()

    def find_elems(self):

        elems_mass_table = pd.read_csv(masses_path)
        main_elems = elems_mass_table['element'].values
        all_elems = elems_mass_table['element_isotop'].values

        elems = []
        for col in self.table.columns:
            if col in main_elems:
                elems.append(col)
            elif col in all_elems:
                elems.append(col)

        return elems

    def load(
            self,
            filename: Union[Path, str],
            mapper: Optional[Mapping[str, str]] = None,
            ignore_columns: Optional[Sequence[str]] = None,
            take_columns: Optional[Sequence[str]] = None,
            take_only_mz: Optional[Sequence[str]] = False,
            sep: str = ",",
            intens_min: Optional[tuple] = None,
            intens_max: Optional[tuple] = None,
            mass_min: Optional[tuple] = None,
            mass_max: Optional[tuple] = None,
            elems: Optional[list] = None,
    ) -> "MassSpectrum":
        self.table = pd.read_csv(filename, sep=sep)
        if mapper:
            self.table = self.table.rename(columns=mapper)

        if take_columns:
            self.table = self.table.loc[:,take_columns]

        if ignore_columns:
            self.table = self.table.drop(columns=ignore_columns)

        if take_only_mz:
            self.table = self.table.loc[:,['mass','intensity']]

        if "numbers" not in self.table:
            self.table["numbers"] = 1

        if intens_min is not None:
            self.table = self.table.loc[self.table['intensity']>intens_min]

        if intens_max is not None:
            self.table = self.table.loc[self.table['intensity']<intens_max]

        if mass_min is not None:
            self.table = self.table.loc[self.table['mass']>mass_min]

        if mass_max is not None:
            self.table = self.table.loc[self.table['mass']<mass_max]

        if elems:
            self.elems = elems
        else:
            self.elems = self.find_elems()

        self.table = self.table.reset_index(drop=True)

        return self

    def save(self, filename: Union[Path, str], sep: str = ",") -> None:
        """Saves to csv MassSpectrum"""
        self.table.to_csv(filename, sep=sep, index=False)

    def assign(
            self,
            elems: Sequence[str] = 'CHONS',
            generated_bruttos_table: pd.DataFrame = None,
            rel_error: float = 0.5,
            sign='-'
    ) -> "MassSpectrum":

        """Finding the nearest mass in generated_bruttos_table

        :param generated_bruttos_table: pandas DataFrame with column 'mass' and elements, should be sorted by 'mass'
        :param elems: Sequence of elements corresponding to generated_bruttos_table
        :param rel_error: error in ppm
        :return: MassSpectra object with assigned signals
        """

        if generated_bruttos_table is None:
            generated_bruttos_table = brutto_gen()

        table = self.table.loc[:,['mass', 'intensity']].copy()

        masses = generated_bruttos_table["mass"].values
        
        if sign == '-':
            mass_shift = - 0.00054858 + 1.007825  # electron and hydrogen mass

        elif sign == '+':
            mass_shift = 0.00054858  # electron mass
        
        else:
            raise ValueError(f"sign can be only + or - rather than {sign}")

        elems = list(generated_bruttos_table.drop(columns=["mass"]))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = []
        for index, row in table.iterrows():
            mass = row["mass"] + mass_shift
            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1

            if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error:
                res.append({**dict(zip(elems, bruttos[idx])), "assign": True})
            else:
                res.append({"assign": False})

        res = pd.DataFrame(res)

        return MassSpectrum(table.join(res), elems=elems)

    def filter_by_C13(
        self, 
        rel_error: float = 0.5,
        remove: bool = False,
        max_charge: int = 1
    ) -> 'MassSpectrum':

        '''
        C13 isotope peak checking
        :param rel_error: allowable ppm error when checking c13 isotope peak
        :param remove: if True peakes without C13 isotopes peak will be dropped
        :param max_charge: max charge in m/z that we looking
        :return: MassSpectra object with cleaned or checked mass-signals
        '''

        table = self.table.sort_values(by='mass').reset_index(drop=True)
        
        flags = np.zeros(table.shape[0], dtype='int')
        masses = table["mass"].values
        
        C13_C12 = 1.003355  # C13 - C12 mass difference

        for z in range(1, max_charge+1):
            for index, row in table.iterrows():
                mass = row["mass"] + C13_C12/z 
                error = mass * rel_error * 0.000001

                idx = np.searchsorted(masses, mass, side='left')
                
                if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                    idx -= 1
                
                if np.fabs(masses[idx] - mass)  <= error:
                    flags[index] = z
        
        table['C13_peak_z'] = flags

        if remove:
            table = table.loc[table['C13_peak_z'] != 0].reset_index(drop=True)

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

        return MassSpectrum(table, elems=list(elems))

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
                   ['intensity', "mass", "brutto", "calculated_mass", "abs_error", "rel_error"] if column in self.table]

        return self.table[columns].__repr__()

    def __str__(self) -> str:
        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__str__()

    def calculate_error(self, sign='-') -> "MassSpectrum":
        if "calculated_mass" not in self.table:
            table = self.calculate_mass()
        else:
            table = self.table.copy()

        if sign == '-':
            table["abs_error"] = table["mass"] - table["calculated_mass"] + (- 0.00054858 + 1.007825) #-electron + proton
        elif sign == '+':
            table["abs_error"] = table["mass"] - table["calculated_mass"] + 0.00054858 #+electron
        else:
            table["abs_error"] = table["mass"] - table["calculated_mass"]
        
        table["rel_error"] = table["abs_error"] / table["mass"] * 1e6

        return MassSpectrum(table)

    def show_error(self):

        fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        ax.scatter(self.table['mass'], self.table['rel_error'], s=0.1)
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('error, ppm')

        return MassSpectrum(self.table)

    def calculate_mass(self) -> "MassSpectrum":
        
        table = self.table.copy()
        table = table.loc[:,self.elems]

        #load elements table. Generatete in mass folder
        elems_mass_table = pd.read_csv(masses_path)
        
        elems_masses = []
        for el in self.elems:
            if '_' not in el:
                temp = elems_mass_table.loc[elems_mass_table['element']==el].sort_values(by='abundance',ascending=False).reset_index(drop=True)
                elems_masses.append(temp.loc[0,'mass'])
            else:
                temp = elems_mass_table.loc[elems_mass_table['element_isotop']==el].reset_index(drop=True)
                elems_masses.append(temp.loc[0,'mass'])

        masses = np.array(elems_masses)
        self.table["calculated_mass"] = table.multiply(masses).sum(axis=1)
        self.table.loc[self.table["calculated_mass"] == 0] = np.NaN

        return MassSpectrum(self.table)

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
        
        if "calculated_mass" not in self.table:
            e = copy.deepcopy(self.calculate_mass())
        else:
            e = copy.deepcopy(self)
        if "calculated_mass" not in other.table:
            s = copy.deepcopy(other.calculate_mass())
        else:
            s = copy.deepcopy(other)

        a = e.table.dropna()
        b = s.table.dropna()
        
        a = a.append(b, ignore_index=True)
        a = a.drop_duplicates(subset=['calculated_mass'])

        return MassSpectrum(a)

    def __xor__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":

        other2 = copy.deepcopy(self)
        sub1 = self.__sub__(other)
        sub2 = other.__sub__(other2)
        
        return sub1.__or__(sub2)

    def __and__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":

        if "calculated_mass" not in self.table:
            e = copy.deepcopy(self.calculate_mass())
        else:
            e = copy.deepcopy(self)
        if "calculated_mass" not in other.table:
            s = copy.deepcopy(other.calculate_mass())
        else:
            s = copy.deepcopy(other)

        a = e.table['calculated_mass'].dropna().values
        b = s.table['calculated_mass'].dropna().values
        
        operate = set(a) & set(b)

        mark = []
        res = copy.deepcopy(self.table)
        for i, row in res.iterrows():
            if row['calculated_mass'] in operate:
                mark.append(row['calculated_mass'])
            else:
                mark.append(np.NaN)
        res['calculated_mass'] = mark
        res = res.dropna()

        return MassSpectrum(res)

    def __add__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        return self.__or__(other)

    def __sub__(self, other):
        
        if "calculated_mass" not in self.table:
            e = copy.deepcopy(self.calculate_mass())
        else:
            e = copy.deepcopy(self)
        if "calculated_mass" not in other.table:
            s = copy.deepcopy(other.calculate_mass())
        else:
            s = copy.deepcopy(other)

        a = e.table['calculated_mass'].dropna().values
        b = s.table['calculated_mass'].dropna().values
        
        operate = set(a) - set(b)

        mark = []
        res = copy.deepcopy(self.table)
        for i, row in res.iterrows():
            if row['calculated_mass'] in operate:
                mark.append(row['calculated_mass'])
            else:
                mark.append(np.NaN)
        res['calculated_mass'] = mark
        res = res.dropna()

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
        table['intensity'] /= table['intensity'].max()
        return MassSpectrum(table)

    def head(self) -> pd.DataFrame:
        return self.table.head()

    def tail(self) -> pd.DataFrame:
        return self.table.tail()

    def draw(self,
             xlim: Tuple[Optional[float], Optional[float]] = (None, None),
             ylim: Tuple[Optional[float], Optional[float]] = (None, None),
             color: str = 'black',
             ax = None,
             name = None
    ) -> None:

        df = self.table.sort_values(by="mass")

        mass = df.mass.values
        if xlim[0] is None:
            xlim = (mass.min(), xlim[1])
        if xlim[1] is None:
            xlim = (xlim[0], mass.max())

        intensity = df['intensity'].values
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

        if ax is None:

            plt.plot(M, I, color=color)
            plt.plot([xlim[0], xlim[1]], [0, 0], color=color)
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel("m/z, Da")
            plt.ylabel("Intensity")
        
        else:
            ax.plot(M, I, color=color, linewidth=0.2)
            ax.plot([xlim[0], xlim[1]], [0, 0], color=color, linewidth=0.2)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel('m/z, Da')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{len(self.table)} peaks\n{name}')

        return

    def intergrate(self, normalize=False) -> Tuple[np.ndarray, np.ndarray]:
        tmp = self.table[["mass"]]
        tmp = tmp.append({"mass": 0}, ignore_index=True).sort_values("mass")
        tmp["index"] = pd.RangeIndex(start=0, stop=len(tmp), step=1)
        if normalize:
            tmp['index'] /= len(tmp)

        return tmp["mass"].values, tmp["index"].values

    def recallibrate(self, error_table, cal_range = None):
        '''
        recallibrate data by error-table
        '''
        err = copy.deepcopy(error_table.table)
        if cal_range is not None:
            err = err.loc[(err['mass']>cal_range[0]) & (err['mass']<cal_range[1])]
        data = self.table.reset_index(drop=True)
        wide = len(err)

        data['old_mass'] = data['mass']

        min_mass = err['mass'].min()
        max_mass = err['mass'].max()
        a = np.linspace(min_mass, max_mass, wide+1)

        for i in range(wide):
            for ind in data.loc[(data['mass']>a[i]) & (data['mass']<a[i+1])].index:
                mass = data.loc[ind, 'mass']
                e = mass * err.loc[i, 'ppm'] / 1000000
                data.loc[ind, 'mass'] = data.loc[ind, 'mass'] + e
                
        return MassSpectrum(data)


class CanNotCreateVanKrevelen(Exception):
    pass


class VanKrevelen(object):
    def __init__(self, table: Optional[Union[pd.DataFrame, 'MassSpectrum']] = None, name: Optional[str] = None):
        self.name = name
        if table is None:
            return

        if isinstance(table, MassSpectrum):
            table = table.table

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

    def draw_scatter(self, ax=None, legend=True, volumes=True, nitrogen=False, sulphur=False, alpha=0.3, mark_elem=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        else:
            ax=ax
        
        if volumes:
            self.table['volume'] = self.table['intensity'] / self.table['intensity'].median()
        else:
            self.table['volume'] = 5

        self.table['color'] = 'blue'

        if mark_elem is not None:
            self.table.loc[self.table[mark_elem] > 0, 'color'] = 'purple'

        if nitrogen and 'N' in self.table.columns:
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] > 0), 'color'] = 'orange'

        if sulphur and 'S' in self.table.columns:
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] < 1) & (self.table['S'] > 0), 'color'] = 'green'
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] > 0) & (self.table['S'] > 0), 'color'] = 'red'

        ax.scatter(self.table["O/C"], self.table["H/C"], s=self.table['volume'], c=self.table['color'], alpha=alpha, **kwargs)
        ax.set_xlabel("O/C")
        ax.set_ylabel("H/C")
        ax.yaxis.set_ticks(np.arange(0, 2.2, 0.4))
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.2)
        
        num_formules = self.table['C'].count()
        ax.set_title(f'{num_formules} formulas\n{self.name}', size=10)

    def get_squares(self, rows: int = 5, columns: int = 4):

        df = self.table
        squares = [None for _ in range(rows * columns)]

        num = 0
        dhc = 2 / rows
        doc = 1 / columns

        for j, oc in zip(np.arange(columns), np.linspace(0, 1, (columns + 1))):
            # for i, hc in zip(np.arange(rows - 1, -1, -1), np.linspace(2.2, 0.2, (rows + 1))):
            for i, hc in zip(np.arange(rows), np.linspace(0.2, 2.2, rows + 1)):
                tmp = MassSpectrum(df[(df["H/C"] >= hc) & (df["H/C"] < (hc + dhc)) & (df["O/C"] >= oc) & (df["O/C"] < (oc + doc))])
                squares[num] = tmp
                num += 1

        return squares


    def get_squares_density(self, rows: int = 5, columns: int = 4, weight: str = "count", dimensions: int = 1):
        squares = self.get_squares(rows=rows, columns=columns)

        density = []

        if weight == "count":

            sum_density = sum([len(squares) for square in squares])
            for square in squares:
                density.append(len(square) / sum_density)

        elif weight == "intensity":
            sum_density = sum([square['intensity'].sum() for square in squares])
            for square in squares:
                density.append(square['intensity'].sum() / sum_density)

        else:
            raise ValueError(f"weight should be count or intensity (not {weight})")

        if dimensions == 1:
            return density

        elif dimensions == 2:
            num = 0
            density2d = np.zeros((rows, columns))
            for column in np.arange(columns):
                for row in np.arange(rows - 1, -1, -1):
                    density2d[row, column] = density[num]
                    num += 1

            return density2d

        else:
            raise ValueError(f"dimensions colud be 1 or 2 (not {dimensions})")

    def get_kellerman_zones(self):

        df = self.table
        C, H, O, N, S = df["C"], df["H"], df["O"], df["N"], df["S"]
        df["H/C"] = H / C
        df["O/C"] = O / C

        # AImod
        df["AI"] = (1 + C - 0.5 * O - 0.5 * H) / (C - 0.5 * O - N)

        AI = df["AI"]
        OC = df["O/C"]
        HC = df["H/C"]

        ans = {}

        ans["lipids"] = MassSpectrum(df[(OC < 0.3) & (HC >= 1.5) & (N == 0)])
        ans["N-satureted"] = MassSpectrum(df[(HC >= 1.5) & (N >= 1)])
        ans["aliphatics"] = MassSpectrum(df[(OC >= 0.3) & (HC >= 1.5) & (N == 0)])

        ans["unsat_lowOC"] = MassSpectrum(df[(HC < 1.5) & (AI < 0.5) & (OC <= 0.5)])
        ans["unsat_highOC"] = MassSpectrum(df[(HC < 1.5) & (AI < 0.5) & (OC > 0.5)])

        ans["aromatic_lowOC"] = MassSpectrum(df[(OC <= 0.5) & (0.5 < AI) & (AI <= 0.67)])
        ans["aromatic_highOC"] = MassSpectrum(df[(OC > 0.5) & (0.5 < AI) & (AI <= 0.67)])

        ans["condensed_lowOC"] = MassSpectrum(df[(OC <= 0.5) & (AI > 0.67)])
        ans["condensed_highOC"] = MassSpectrum(df[(OC > 0.5) & (AI > 0.67)])

        return ans

    def get_kellerman_density(self, weight: str = "count", return_keys=False):
        kellerman = self.get_kellerman_zones()
        ans = {}

        if weight == "count":
            sum_density = len(self.table)
            for key, value in kellerman.items():
                ans[key] = len(value.table) / sum_density

        elif weight == "intensity":
            sum_density = self.table['intensity'].sum()
            for key, value in kellerman.items():
                ans[key] = value.table['intensity'].sum() / sum_density

        else:
            raise ValueError(f"weight should be count or intensity not {weight}")

    def boxed_van_krevelen(self, r=5, c=4) -> Sequence[Sequence]:
        # (array([0.2, 0.6, 1. , 1.4, 1.8, 2.2]), array([0.  , 0.25, 0.5 , 0.75, 1.  ]))

        df = self.table
        y = np.linspace(0.2, 2.2, r + 1)  # 0.4
        x = np.linspace(0, 1, c + 1)  # 0.25

        vc = []
        for i in range(c):
            vc.append([])
            for j in range(r):
                vc[-1].append(
                    df[
                        (df["H/C"] > y[j]) & (df["H/C"] <= y[j] + 2.0 / r) &
                        (df["O/C"] > x[i]) & (df["O/C"] <= x[i] + 1.0 / c)
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

    def plot_heatmap(self, df):

        fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        sns.heatmap(df.round(4),cmap='coolwarm',annot=True, linewidths=.5, ax=ax)
        bottom, top = ax.get_ylim()
        plt.yticks(rotation=0)
        plt.xticks(rotation=90) 
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')
        fig.tight_layout()

    def squares(self):
        d_table = []
        total_i = self.table['intensity'].sum()
        for y in [ (1.8, 2.2), (1.4, 1.8), (1, 1.4), (0.6, 1), (0, 0.6)]:
            hc = []
            for x in  [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]:
                temp = copy.deepcopy(self)
                temp.table = temp.table.loc[(temp.table['O/C'] > x[0]) & (temp.table['O/C'] < x[1]) & (temp.table['H/C'] > y[0]) & (temp.table['H/C'] < y[1])]
                temp_i = temp.table['intensity'].sum()
                hc.append(temp_i/total_i)
            d_table.append(hc)
        out = pd.DataFrame(data = d_table, columns=['0-0.25', '0,25-0.5','0.5-0.75','0.75-1'], index=['1.8-2.2', '1.4-1.8', '1-1.4', '0.6-1', '0-0.6'])
        self.plot_heatmap(out)
        
        return out

    @staticmethod
    def save_fig(path, dpi=300) -> None:
        """
        Save picture
        Be careful! If axes are used, it can work incorrect!

        :param path:
        :return:
        """
        plt.savefig(path, dpi=dpi)

    @staticmethod
    def show():
        """
        This method is needed to hide plt
        Sometimes we don't want to use additional imports
        :return:
        """
        plt.show()

    def save(self, path: Union[Path, str], sep: str = ';') -> None:
        """
        Saves VK to the table with path
        :param path: filename should have extension
        :param sep:
        :return:
        """
        self.table.to_csv(path, sep=sep, index=False)

    @staticmethod
    def load(path, sep=';') -> 'VanKrevelen':
        """
        Loads VK from table, name is the filename without extension
        :param path:
        :param sep:
        :return:
        """
        table = pd.read_csv(path, sep=sep)
        name = ".".join(str(path).split("/")[-1].split(".")[:-1])

        return VanKrevelen(table=table, name=name)


def calculate_ppm(x: float, y: float) -> float:
    return np.fabs((x - y) / y * 1e6)


class ErrorTable(object):
    # should be columns: mass, ppm

    def __init__(
            self,
            table: Optional[pd.DataFrame] = None,
    ):
        self.table = table

    def dif_mass(self):
        '''
        most common mass difference
        return dict
        '''
        H = 1.007825
        C = 12.000000
        N = 14.003074
        O = 15.994915
        S = 31.972071

        dif = {}
        dif['CH2'] = C + H*2
        dif['CH2O'] = C + O + H*2
        dif['C2H2O'] = C*2 + O + H*2
        dif['CO2'] = C + O*2
        dif['H2O'] = O + H*2

        return dif

    def md_error_map (self, spec, ppm=3, show_map=True):
        '''
        calculate mass differnce map
        '''

        dif = self.dif_mass()
        data = copy.deepcopy(spec)
        data = data.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        data = data[:1000]
        data = data.sort_values(by='mass').reset_index(drop=True)

        data_error = [] #array for new data

        for i in range(len(data)): #take every mass in list
            mass = data.loc[i, 'mass'] #take mass from list
            for k in range(1,6): #generation of brutto series change
                for i in dif: #take most common mass diff
                    mz = mass + dif[i]*k #calc mass plus ion
                    mz_p = mz + mz * ppm/1000000 #search from min
                    mz_m = mz - mz * ppm/1000000 # to max
                    ress = data.loc[(data['mass'] > mz_m) & (data['mass'] < mz_p)]
                    if len(ress) > 0:
                        res = copy.deepcopy(ress)
                        res['ppm'] = (((res['mass'] - mz) / mz)*1000000).abs()
                        min_ppm = res['ppm'].min()
                        min_mz = res.loc[res['ppm']==min_ppm, 'mass'].values[0]
                        data_error.append([mass, f'{k}*{i}', min_mz, (min_mz-mz)/mz*1000000])
        
        df_error = pd.DataFrame(data = data_error, columns=['mass', 'mass_diff_brutto', 'mass_diff_mass', 'ppm' ])
        
        if show_map:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            ax.scatter(df_error['mass'], df_error['ppm'], s=0.01)

        return df_error
    
    def fit_kernel(self, f, show_map=True):
        '''
        fit max intesity of kernel density map
        return error table for 100 values
        '''
        df = pd.DataFrame(f, index=np.linspace(3,-3,100))
        
        out = []
        for i in df.columns:
            max_kernel = df[i].quantile(q=0.95)
            ppm = df.loc[df[i] > max_kernel].index.values
            out.append([i, np.mean(ppm)])
        kde_err = pd.DataFrame(data=out, columns=['i','ppm'])
        
        #smooth data
        kde_err['ppm'] = savgol_filter(kde_err['ppm'], 31,5)

        xmin = 0
        xmax = 100
        ymin = -3
        ymax = 3

        if show_map:
            fig = plt.figure(figsize=(4,4), dpi=75)
            ax = fig.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.imshow(df, extent=[xmin, xmax, ymin, ymax], aspect='auto')
            ax.plot(kde_err['i'], kde_err['ppm'], c='r')

        #lock start at zero
        kde_err['ppm'] = kde_err['ppm'] - kde_err.loc[0,'ppm']
        return kde_err

    def kernel_density_map(self, df_error, ppm=3, show_map=False):
        '''
        plot kernel density map 100*100 for data
        '''
        
        x = np.array(df_error['mass'])
        y = np.array(df_error['ppm'])

        xmin = min(x) 
        xmax = max(x) 

        ymin = -ppm 
        ymax = ppm 

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f = np.rot90(f)

        if show_map:
            fig = plt.figure(figsize=(4,4), dpi=75)
            ax = fig.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.imshow(f, extent=[xmin, xmax, ymin, ymax], aspect='auto')
        
        return f

    def assign_error(self, spec:MassSpectrum,
                    ppm = 3,
                    sign = '-',
                   show_map:bool = True):
        '''
        recallibrate by assign
        '''
        spectr = copy.deepcopy(spec)
        spectr = spectr.assign(rel_error=ppm) 
        spectr = spectr.calculate_mass()
        spectr = spectr.calculate_error(sign=sign)
        spectr = spectr.show_error()

        error_table = spectr.table
        error_table = error_table.loc[:,['mass','rel_error']]
        error_table.columns = ['mass', 'ppm']
        error_table = error_table.dropna()

        kde = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kde, show_map=True)

        err['ppm'] = - err['ppm']
        err['mass'] = np.linspace(error_table['mass'].min(), error_table['mass'].max(),len(err))

        return ErrorTable(err)  

    def massdiff_error(self,
                                spec:MassSpectrum,
                                show_map = True):
        '''
        self-recallibration of mass-spectra by mass-difference map
        take a lot of time
        based on work:
        Smirnov, K. S., Forcisi, S., Moritz, F., Lucio, M., & Schmitt-Kopplin, P. 
        (2019). Mass difference maps and their application for the 
        recalibration of mass spectrometric data in nontargeted metabolomics. 
        Analytical chemistry, 91(5), 3350-3358. 
        '''
        spec_table = copy.deepcopy(spec.table)
        mde = self.md_error_map(spec = spec_table, show_map=show_map)
        f = self.kernel_density_map(df_error=mde)
        err = self.fit_kernel(f=f, show_map=show_map)
        err['mass'] = np.linspace(spec.table['mass'].min(), spec.table['mass'].max(),len(err))

        return ErrorTable(err)

    def etalon_error( self,
                    spec, #initial masspectr
                    etalon, #etalon massspectr
                    quart=0.9, #treshold by quartile
                    ppm=3,#treshold by ppm
                    show_error=True
                    ): 
        '''
        recallibrate by etalon
        '''

        et = copy.deepcopy(etalon.table)['mass'].to_list()
        df = copy.deepcopy(spec.table)

        min_mass = df['mass'].min()
        max_mass = df['mass'].max()
        a = np.linspace(min_mass,max_mass,101)

        treshold = df['intensity'].quantile(quart)
        df = df.loc[df['intensity'] > treshold].reset_index(drop = True)
        df['cal'] = 0 #column for check

        #fill data massiv with correct mass
        for i in range(0,len(df)):
            min_mass = df.loc[i, 'mass']*(1 - ppm/1000000)
            max_mass = df.loc[i, 'mass']*(1 + ppm/1000000)
            for mass in et:
                try:
                    if mass > min_mass and mass < max_mass:
                        df.loc[i, 'cal'] = mass
                except:
                    pass
        
        # take just assigned peaks
        df = df.loc[df['cal']>0]
        #calc error and mean error
        df['dif'] = df['cal'] - df['mass']
        mean_e = df['dif'].mean()

        #make error table
        cor = []
        for i in range(0,100):
            correct = df.loc[(df['mass'] > a[i]) & (df['mass'] < a[i+1])]['dif'].mean()
            cor.append((a[i], correct))

        #out table
        err = pd.DataFrame(data=cor, columns=['m/z', 'err'])
        err['err'] = err['err'].fillna(mean_e)
        err['ppm']=err['err']/err['m/z']*1000000

        err['ppm'] = savgol_filter(err['ppm'], 51,5)
        err['mass'] = np.linspace(df['mass'].min(), df['mass'].max(),len(err))

        if show_error:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            ax.plot(err['m/z'], err['ppm'])
            ax.set_xlabel('m/z, Da')
            ax.set_ylabel('Error, ppm')
            fig.tight_layout()
        
        return ErrorTable(err)

    def extrapolate(self, ranges=(200,900)):
        """
        extrapolate exsist error data for disire diapason, make 100 points
        param:ranges - tuple, take diapason for extrapolate. Default 200-900
        """
        
        interpolation_range = np.linspace(ranges[0], ranges[1], 100)
        linear_interp = interp1d(self.table['mass'], self.table['ppm'],  bounds_error=False, fill_value='extrapolate')
        linear_results = linear_interp(interpolation_range)
        err = pd.DataFrame()
        err ['mass'] = interpolation_range
        err ['ppm'] = linear_results

        return ErrorTable(err)

    def show_error(self):
        
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)
        ax.plot(self.table['mass'], self.table['ppm'])
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('error, ppm')


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
                    table[-1].append(spectrum['intensity'].values[idx])
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
                vector.append(spectrum[brutto]['intensity'] if brutto in spectrum else 0)

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
