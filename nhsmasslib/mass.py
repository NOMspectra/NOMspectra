#    Copyright 2019-2021 Rukhovich Gleb
#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nhsmasslib. 
#
#    nhsmasslib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nhsmasslib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nhsmasslib.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from typing import Sequence, Union, Optional, Mapping, Tuple
import copy

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import scipy.stats as st
from scipy import spatial
from scipy.optimize import curve_fit

from tqdm import tqdm

from .brutto import brutto_gen, elements_table, get_elements_masses


class MassSpectrum(object):
    """ 
    A class used to represent mass spectrum

    Attributes
    ----------
    table : pandas Datarame
        Optional. consist spectrum (mass and intensity of peaks) and all calculated parameters
        like brutto formulas, calculated mass, relative errorr
    elems : list
        Optional. Consist elements that used for mass spectrum treatment
        can be finded by class method find_elems()
    """

    def __init__(
                self,
                table: pd.DataFrame = None,
                elems: Sequence[str] = None,
                ) -> pd.DataFrame:
        """
        Parameters
        ----------
        table : pandas Datarame
            Optional. Consist spectrum (mass and intensity of peaks) and all calculated 
            parameters like brutto formulas, calculated mass, relative errorr
        elems : list
            Optional. Consist elements that used for mass spectrum treatment
            can be finded by class method find_elems()
        """

        self.features = ["mass", "calculated_mass", 'intensity', "abs_error", "rel_error"]

        if table is not None:
            self.table = table
        else:
            self.table = pd.DataFrame(columns=['intensity', "mass", "brutto", "calculated_mass", "abs_error", "rel_error"])

        if elems is not None:
            self.elems = elems
        else:
            self.elems = self.find_elems()

    def find_elems(self) -> Sequence[str]:
        """ 
        Find elems from mass spectrum table.

        Find elements in table columns. Used elems_mass_table with all elements and isotopes.
        For example, element 'C' will be recognised as carbon 12C, element 'C_13" as 13C

        Returns
        -------
        list of found elemets in columns label. For example: ['C','H','O','N']
        """

        main_elems = elements_table()['element'].values
        all_elems = elements_table()['element_isotop'].values

        elems = []
        for col in self.table.columns:
            if col in main_elems:
                elems.append(col)
            elif col in all_elems:
                elems.append(col)

        if len(elems) == 0:
            elems = None

        return elems

    def load(
        self,
        filename: Union[Path, str],
        mapper: Mapping[str, str] = None,
        ignore_columns: Sequence[str] = None,
        take_columns: Sequence[str] = None,
        take_only_mz: Sequence[str] = False,
        sep: str = ",",
        intens_min: float =  None,
        intens_max: float = None,
        mass_min: float =  None,
        mass_max: float = None,
    ) -> "MassSpectrum":
        """
        Load mass pectrum table to MassSpectrum object

        All parameters is optional except filename

        Parameters
        ----------
        filename: str
            path to mass spectrum table, absoulute or relative
        mapper: dict
            dictonary for recognize columns in mass spec file. 
            Example: {'m/z':'mass','I':'intensity'}
        ignore_columns: list of str
            list with names of columns that willn't loaded.
            if None load all columns.
            Example: ["index", "s/n"]
        take_columns: list of str
            list with names of columns that only will be loaded.
            if None load all columns.
            Example: ["mass", "intensity", "C", "H", "N", "O"]
        take_only_mz: bool
            load only mass and intesivity columns
        sep: str
            separator in mass spectrum table, \\t - for tab.
        intens_min: numeric
            bottom limit for intensivity.
            by default None and don't restrict by this.
            But for some spectrum it is necessary to cut noise.
        intens_max: numeric
            upper limit for intensivity.
            by default None and don't restrict by this
        mass_min: numeric
            bottom limit for m/z.
            by default None and don't restrict by this
        mass_max: numeric
            upper limit for m/z.
            by default None and don't restrict by this

        Return
        ------
        MassSpectrum object
        """

        self.table = pd.read_csv(filename, sep=sep)
        if mapper:
            self.table = self.table.rename(columns=mapper)

        if take_columns:
            self.table = self.table.loc[:,take_columns]

        if ignore_columns:
            self.table = self.table.drop(columns=ignore_columns)

        if take_only_mz:
            self.table = self.table.loc[:,['mass','intensity']]

        if intens_min is not None:
            self.table = self.table.loc[self.table['intensity']>intens_min]

        if intens_max is not None:
            self.table = self.table.loc[self.table['intensity']<intens_max]

        if mass_min is not None:
            self.table = self.table.loc[self.table['mass']>mass_min]

        if mass_max is not None:
            self.table = self.table.loc[self.table['mass']<mass_max]

        self.elems = self.find_elems()

        if self.elems is not None:
            self._mark_assigned_by_brutto()

        self.table = self.table.reset_index(drop=True)

        return self

    def _mark_assigned_by_brutto(self) -> None:
        """Mark paeks in loaded mass list if they have brutto

        Return
        ------
        MassSpectrum object with assigned mark
        """

        assign = []
        for i, row in self.table.iterrows():
            flag = False
            for el in self.elems:
                if row[el] > 0:
                    flag = True
            assign.append(flag) 
        self.table['assign'] = assign

    def save(self, filename: Union[Path, str], sep: str = ",") -> None:
        """
        Saves to csv MassSpectrum
        
        Parameters
        ----------
        filename: str
            Path for saving mass spectrum table with calculation to csv file
        sep: str
            Optional. Separator in saved file. By default it is ','        
        """
        self.table.to_csv(filename, sep=sep, index=False)

    def assign(
            self,
            generated_bruttos_table: pd.DataFrame = None,
            rel_error: float = 0.5,
            sign: str ='-'
    ) -> "MassSpectrum":
        """
        Finding the nearest mass in generated_bruttos_table
        
        Parameters
        -----------
        generated_bruttos_table: pandas DataFrame 
            Optional. Contain column 'mass' and elements, 
            should be sorted by 'mass'.
            Can be generated by function brutto_generator.brutto_gen(). 
            if 'None' generate table with default elemnets and ranges
            C: 4-50, H 4-100, O 0-25, N 0-3, S 0-2.
        rel_error: float
            Optional. default 0.5, permissible error in ppm for assign mass to brutto formulas
        sign: str
            Optional. Deafult '-'.
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            None for neutral

        Return
        ------
        MassSpectra object with assigned signals
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
            mass_shift = 0

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
    ) -> 'MassSpectrum':
        """ 
        C13 isotope peak checking

        Parameters
        ----------
        rel_error: float
            Optional. Default 0.5.
            Allowable ppm error when checking c13 isotope peak
        remove: bool
            Optional, default False. 
            if True peakes without C13 isotopes peak will be dropped
        
        Return
        ------
        MassSpectra object with cleaned or checked mass-signals
        """

        table = self.table.sort_values(by='mass').reset_index(drop=True)
        
        flags = np.zeros(table.shape[0], dtype=bool)
        masses = table["mass"].values
        
        C13_C12 = 1.003355  # C13 - C12 mass difference

        
        for index, row in table.iterrows():
            mass = row["mass"] + C13_C12
            error = mass * rel_error * 0.000001

            idx = np.searchsorted(masses, mass, side='left')
            
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1
            
            if np.fabs(masses[idx] - mass)  <= error:
                flags[index] = True
        
        table['C13_peak'] = flags

        if remove:
            table = table.loc[(table['C13_peak'] == True) & (table['assign'] == True)].reset_index(drop=True)

        return MassSpectrum(table)

    def calculate_brutto(self) -> 'MassSpectrum':
        """
        Calculate brutto formulas from assign table

        Return
        ------
        MassSpectrum object wit calculated bruttos
        """

        table = copy.deepcopy(self.table)

        elems = self.find_elems()
        out = []
        for i, row in table.iterrows():
            s = ''
            for el in elems:
                if row[el] == 1:
                    s = s + f'{el}'
                elif row[el] > 0:
                    s = s + f'{el}{int(row[el])}'
            out.append(s)
        
        table['brutto'] = out

        return MassSpectrum(table)

    def copy(self) -> 'MassSpectrum':
        """
        Deepcopy of self MassSpectrum object

        Return
        ------
        Deepcopy of self MassSpectrum object
        """
        return copy.deepcopy(MassSpectrum(self.table))

    def calculate_error(self, sign: str ='-') -> "MassSpectrum":
        """
        Calculate relative and absolute error of assigned peaks

        Parameters
        ----------
        sign: str
            Optional. Default '-'. 
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            None for neutral
        
        Return
        ------
        MassSpectrum object wit calculated error
        """
        if "calculated_mass" not in self.table:
            table = self.calculate_mass().table
        else:
            table = copy.deepcopy(self.table)

        if sign == '-':
            table["abs_error"] = table["mass"] - table["calculated_mass"] + (- 0.00054858 + 1.007825) #-electron + proton
        elif sign == '+':
            table["abs_error"] = table["mass"] - table["calculated_mass"] + 0.00054858 #+electron
        else:
            table["abs_error"] = table["mass"] - table["calculated_mass"]
        
        table["rel_error"] = table["abs_error"] / table["mass"] * 1e6

        return MassSpectrum(table)

    def show_error(self) -> None:
        """
        Plot relative error of assigned brutto formulas vs mass
        """

        if "rel_error" not in self.table:
            self = self.calculate_error()      

        fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        ax.scatter(self.table['mass'], self.table['rel_error'], s=0.1)
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('error, ppm')

    def calculate_mass(self) -> "MassSpectrum":
        """
        Calculate mass from assigned brutto formulas

        Return
        ------
        MassSpectrum object with calculated mass
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")
        
        table = copy.deepcopy(self.table)
        self.elems = self.find_elems()
        
        table = table.loc[:,self.elems]
        
        masses = get_elements_masses(self.elems)

        self.table["calculated_mass"] = table.multiply(masses).sum(axis=1)
        self.table["calculated_mass"] = np.round(self.table["calculated_mass"], 6)
        self.table.loc[self.table["calculated_mass"] == 0, "calculated_mass"] = np.NaN

        return MassSpectrum(self.table)
    
    def __repr__(self) -> str:
        """
        Representation of MassSpectrum object.

        Return
        ------
        the string representation of MassSpectrum object
        """

        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__repr__()

    def __str__(self) -> str:
        """
        Representation of MassSpectrum object.

        Return
        ------
        the string representation of MassSpectrum object
        """
        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__str__()

    def __or__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        """
        Logic function or for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain all assigned brutto formulas from two spectrum
        """
        
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
        """
        Logic function xor for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain xor assigned brutto formulas from two spectrum
        """

        other2 = copy.deepcopy(self)
        sub1 = self.__sub__(other)
        sub2 = other.__sub__(other2)
        
        return sub1.__or__(sub2)

    def __and__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        """
        Logic function and for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain common assigned brutto formulas from two spectrum
        """

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
        """
        Logic function or for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain all assigned brutto formulas from two spectrum
        """
        return self.__or__(other)

    def __sub__(self, other):
        """
        Logic function substraction for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain substraction assigned brutto formulas from two spectrum
        """
        
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

    def intens_sub(self, other:"MassSpectrum") -> "MassSpectrum":
        """
        Calculate substruction by intensivity

        Parameters
        ----------
        other: MassSpectrum object
            other mass-scpectrum

        Return
        ------
        MassSpectrum object contain only that peak
        that higher than in other. And intensity of this peaks
        is substraction of self and other.
        """
        #find common masses
        m = self & other
        msc = m.table['calculated_mass'].values

        #extract table with common masses
        massE = self.table['calculated_mass'].values
        rE = self.table[np.isin(massE, msc)]
        massL = other.table['calculated_mass'].values
        rL = other.table[np.isin(massL, msc)]

        #substract intensity each others
        rE = rE.copy()
        rE['intensity'] = rE['intensity'] - rL['intensity']
        rE = rE.loc[rE['intensity'] > 0]
        
        #and add only own molecules
        return (self - other) + MassSpectrum(rE)  

    def __len__(self) -> int:
        """
        Length of Mass-Spectrum table

        Return
        ------
        int - length of Mass-Spectrum table
        """
        return len(self.table)
    
    def __getitem__(self, item: Union[str, Sequence[str]]) -> pd.DataFrame:
        """
        Get items or slice from spec

        Return
        ------
        Pandas Dataframe or Series slices
        """

        return self.table[item]

    def drop_unassigned(self) -> "MassSpectrum":
        """
        Drop unassigned mass from Mass Spectrum table

        Return
        ------
        MassSpectrum object that contain only assigned by brutto formulas peaks

        Caution
        -------
        Danger of lose data - with these operation we exclude data that can be usefull
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        return MassSpectrum(self.table.loc[self.table["assign"] == True].reset_index(drop=True))

    def calculate_simmilarity(self, other:"MassSpectrum", mode:str='cosine') -> float:
        """
        Calculate Simmilarity

        Parameters
        ----------
        other: MassSpectrum object
            second MaasSpectrum object with that calc simmilarity
        mode: str
            Optionaly. Default cosine. 
            one of the similarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"

        Return
        ------
        float Simmilarity index
        """

        if 'calculated_mass' not in self.table:
            self = self.calculate_mass()
        if 'calculated_mass' not in other.table:
            other = other.calculate_mass()

        a = self.table['calculated_mass'].dropna().values
        b = other.table['calculated_mass'].dropna().values
        c = np.union1d(a, b)

        A = np.zeros(len(c), dtype=bool)
        B = np.zeros(len(c), dtype=bool)
        for i, el in enumerate(c):
            if el in a:
                A[i] = True
            if el in b:
                B[i] = True

        if mode == "jaccard":
            return 1 - spatial.distance.jaccard(A, B)
        elif mode == "tanimoto":
            return 1 - spatial.distance.rogerstanimoto(A, B)
        elif mode == 'cosine':
            return 1 - spatial.distance.cosine(A, B)
        else:
            raise Exception(f"There is no such mode: {mode}")

    def calculate_cram(self) -> "MassSpectrum":
        """
        Calculate if include into CRAM
        (carboxylic-rich alicyclic molecules)

        Return
        ------
        MassSpectrun object with check CRAM (bool)

        Reference
        ---------
        Hertkorn, N. et al. Characterization of a major 
        refractory component of marine dissolved organic matter.
        Geochimica et. Cosmochimica Acta 70, 2990-3010 (2006)
        """
        spec = self.copy()
        if "DBE" not in spec.table:
            spec = spec.calculate_dbe()

        def check(row):
            if row['DBE']/row['C'] < 0.3 or row['DBE']/row['C'] > 0.68:
                return False
            if row['DBE']/row['H'] < 0.2 or row['DBE']/row['H'] > 0.95:
                return False
            if row['O'] == 0:
                False
            elif row['DBE']/row['O'] < 0.77 or row['DBE']/row['O'] > 1.75:
                return False
            return True

        spec.table['CRAM'] = spec.table.apply(check, axis=1)

        return spec

    def get_cram_value(self) -> int:
        """
        Calculate percent of CRAM molecules
        (carboxylic-rich alicyclic molecules)

        Return
        ------
        int. percent of CRAM molecules in mass-spec
        weight by intensity
        """
        spec = self.copy()
        if "CRAM" not in spec.table:
            spec = spec.calculate_cram().drop_unassigned()

        value = spec.table.loc[spec.table['CRAM'] == True, 'intensity'].sum()/spec.table['intensity'].sum()
        return int(value*100)

    def calculate_ai(self) -> 'MassSpectrum':
        """
        Calculate AI

        Return
        ------
        MassSpectrum object with calculated AI
        """
        table = self.calculate_cai().calculate_dbe_ai().table
        table["AI"] = table["DBE_AI"] / table["CAI"]

        return MassSpectrum(table)

    def calculate_cai(self) -> 'MassSpectrum':
        """
        Calculate CAI

        Return
        ------
        MassSpectrum object with calculated CAI
        """
        
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = copy.deepcopy(self.table)

        # very careful
        # anyway it's necessary to have at least column with C?
        for element in "CONSP":
            if element not in table:
                table[element] = 0

        self.table['CAI'] = table["C"] - table["O"] - table["N"] - table["S"] - table["P"]

        return self

    def calculate_dbe_ai(self) -> 'MassSpectrum':
        """
        Calculate DBE

        Return
        ------
        MassSpectrum object with calculated DBE
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = copy.deepcopy(self.table)

        for element in "CHONPS":
            if element not in table:
                table[element] = 0

        self.table['DBE_AI'] = 1.0 + table["C"] - table["O"] - table["S"] - 0.5 * (table["H"] + table['N'] + table["P"])

        return self

    def calculate_dbe(self) -> 'MassSpectrum':
        """
        Calculate DBE

        Return
        ------
        MassSpectrum object with calculated DBE
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = copy.deepcopy(self.table)

        for element in "CHON":
            if element not in table:
                table[element] = 0

        self.table['DBE'] = 1.0 + table["C"] - 0.5 * (table["H"] - table['N'])

        return self

    def normalize(self, how:str='sum') -> 'MassSpectrum':
        """
        Intensity normalize by max intensity

        Parameters
        ----------
        how: str
            two option: 
            'max' for normilize by maximum peak.
            'sum' for normilize by sum of intensity of all peaks. (default)

        Return
        ------
        Intensity normalized MassSpectrum instance
        """
        table = self.table.copy()
        if how=='max':
            table['intensity'] /= table['intensity'].max()
        elif how=='sum':
            table['intensity'] /= table['intensity'].sum()
        else:
            Exception(f"There is no such mode: {how}")

        return MassSpectrum(table)

    def head(self, num:int = None) -> pd.DataFrame:
        """
        Show head of mass spec table

        Parameters
        ----------
        num: int
            Optional. number of head string

        Return
        ------
        Pandas Dataframe head of MassSpec table
        """
        if num is None:
            return self.table.head()
        else:
            return self.table.head(num)

    def tail(self, num:int = None) -> pd.DataFrame:
        """
        Show tail of mass spec table

        Parameters
        ----------
        num: int
            Optional. number of tail string

        Return
        ------
        Pandas Dataframe tail of MassSpec table
        """
        if num is None:
            return self.table.tail()
        else:
            return self.table.tail(num)

    def draw(self,
        xlim: Tuple[Optional[float], Optional[float]] = (None, None),
        ylim: Tuple[Optional[float], Optional[float]] = (None, None),
        color: str = 'black',
        ax: plt.axes = None,
        ) -> None:
        """
        Draw mass spectrum

        All parameters is optional

        Parameters
        ----------
        xlim: Tuple (float, float)
            restrict for mass
        ylim: Tuple (float, float)
            restrict for intensity
        color: str
            color of draw
        ax: matplotlyp axes object
            send here ax to plot in your own condition
        """

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
            fig, ax = plt.subplots(figsize=(4,4), dpi=75)
    
        ax.plot(M, I, color=color, linewidth=0.2)
        ax.plot([xlim[0], xlim[1]], [0, 0], color=color, linewidth=0.2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('Intensity')
        ax.set_title(f'{len(self.table)} peaks')

        return

    def recallibrate(self, error_table: "ErrorTable" = None, how = 'assign') -> "MassSpectrum":
        '''
        Recallibrate data by error-table

        Parameters
        ----------
        error_table: ErrorTable object
            Optional. If None - calculate for self. 
            ErrorTable object contain table error in ppm for mass, default 100 string            

        how: str
            Optional. Default 'assign'.
            If error_table is None we can choose how to recalculate.
            'assign' - by assign error, default.
            'mdm' - by calculation mass-difference map.
            filename - path to etalon spectrum, treated and saved by masslib

        Returns
        -------
        MassSpectrum object with recallibrated mass
        '''
        if error_table is None:
            if how == 'assign':
                if "assign" not in self.table:
                    raise Exception("Spectrum is not assigned")
                error_table = ErrorTable().assign_error(self).zeroshift(self)
            elif how == 'mdm':
                error_table = ErrorTable().massdiff_error(self)
            else:
                etalon = MassSpectrum().load(filename=how)
                error_table = ErrorTable().etalon_error(spec=self, etalon=etalon)

        err = copy.deepcopy(error_table.table)
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

    def assign_by_tmds (
        self, 
        tmds_spec: "Tmds" = None, 
        abs_error: float = 0.001,
        p = 0.2,
        max_num: int = None,
        C13_filter: bool = True
        ) -> "MassSpectrum":
        '''
        Assigne brutto formulas by TMDS

        Parameters
        ----------
        tmds_spec: Tmds object
            Optional. if None generate tmds spectr with default parameters
            Tmds object, include table with most probability mass difference
        abs_error: float
            Optional, default 0.001. Error for assign peaks by massdif
        p: float
            Optional. Default 0.2. 
            Relative probability coefficient for treshold tmds spectrum
        max_num: int
            Optional. Max mass diff numbers
        C13_filter: bool
            Use only peaks with C13 isotope peak for generate tmds

        Return
        ------
        MassSpectrum object new assign brutto formulas
        '''
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        if tmds_spec is None:
            tmds_spec = Tmds().calc(self, p=p, C13_filter=C13_filter) #by varifiy p-value we can choose how much mass-diff we will take
            tmds_spec = tmds_spec.assign(max_num=max_num)
            tmds_spec = tmds_spec.calculate_mass()

        tmds = tmds_spec.table.sort_values(by='probability', ascending=False).reset_index(drop=True)
        tmds = tmds.loc[tmds['probability'] > p]
        elem = tmds_spec.elems

        spec = copy.deepcopy(self)
        
        assign_false = copy.deepcopy(spec.table.loc[spec.table['assign'] == False]).reset_index(drop=True)
        assign_true = copy.deepcopy(spec.table.loc[spec.table['assign'] == True]).reset_index(drop=True)
        masses = assign_true['mass'].values
        mass_dif_num = len(tmds)

        for i, row_tmds in tqdm(tmds.iterrows(), total=mass_dif_num):

            mass_shift = - row_tmds['calculated_mass']
            
            for index, row in assign_false.iterrows():
                if row['assign'] == True:
                    continue
                     
                mass = row["mass"] + mass_shift
                idx = np.searchsorted(masses, mass, side='left')
                if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                    idx -= 1

                if np.fabs(masses[idx] - mass) <= abs_error:
                    assign_false.loc[index,'assign'] = True
                    for el in elem:
                        assign_false.loc[index,el] = row_tmds[el] + assign_true.loc[idx,el]

        assign_true = assign_true.append(assign_false, ignore_index=True).sort_values(by='mass').reset_index(drop=True)

        out = MassSpectrum(assign_true)
        out = out.calculate_error()
        
        out_false = out.table.loc[out.table['assign'] == False]
        out_true = out.table.loc[out.table['assign'] == True].drop_duplicates(subset="calculated_mass")

        out2 = pd.merge(out_true, out_false, how='outer').reset_index(drop=True).sort_values(by='mass').reset_index(drop=True)
        
        return MassSpectrum(out2)

    def calculate_DBEvsO(self, ax=None, olim=None, **kwargs) -> None:
        """
        Draw plot DBE by nO and calculate linear fit
        
        Parameters
        ----------
        ax: matplotlib axes
            ax fo outer plot. Default None
        olim: tuple of two int
            limit for nO. Deafult None
        **kwargs: dict
            dict for additional condition to scatter matplotlib

        References
        ----------
        Bae, E., Yeo, I. J., Jeong, B., Shin, Y., Shin, K. H., & Kim, S. (2011). 
        Study of double bond equivalents and the numbers of carbon and oxygen 
        atom distribution of dissolved organic matter with negative-mode FT-ICR MS.
        Analytical chemistry, 83(11), 4193-4199.
        
        """

        spec = self.copy().calculate_dbe().drop_unassigned()
        if olim is None:
            no = list(range(5, int(spec.table['O'].max())-4))
        else:
            no = list(range(olim[0],olim[1]))

        dbe_o = []
        
        for i in no:
            dbes = spec.table.loc[spec.table['O'] == i, 'DBE']
            intens = spec.table.loc[spec.table['O'] == i, 'intensity']
            dbe_o.append((dbes*intens).sum()/intens.sum())
    
        def linear(x, a, b):
            return a*x + b

        x = np.array(no)
        y = np.array(dbe_o)

        popt, pcov = curve_fit(linear, x, y)
        residuals = y- linear(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if ax is None:
            fig,ax = plt.subplots(figsize=(3,3), dpi=100)
        
        ax.scatter(x, y, **kwargs)
        ax.plot(x, linear(x, *popt), label=f'y={round(popt[0],2)}x + {round(popt[1],1)} R2={round(r_squared, 4)}', **kwargs)
        ax.set_xlim(4)
        ax.set_ylim(5)
        ax.set_xlabel('number of oxygen')
        ax.set_ylabel('DBE average')
        ax.legend()


class VanKrevelen(object):
    """
    A class used to plot Van-Krevelen diagramm

    Attributes
    ----------
    table : pandas Datarame or MassSpectrum object
        consist spectrum (mass and intensity of peaks) and all calculated parameters.
        Must contain elements 'C', 'H', 'N'
    """

    def __init__(self, table: Optional[Union[pd.DataFrame, 'MassSpectrum']] = None) -> None:
        """
        Init and calculate C/H, O/C relatives

        Parameters
        ----------
        table : pandas Datarame or MassSpectrum object
            consist spectrum (mass and intensity of peaks) and all calculated parameters
            Must contain elements 'C', 'H', 'N'
        """
        if table is None:
            return

        if isinstance(table, MassSpectrum):
            table = table.table

        if not (("C" in table and "H" in table and "O" in table) or ("O/C" in table or "H/C" in table)):
            raise Exception("There are not H, C, O in table")

        table = table.loc[table["C"] > 0]

        self.table = table
        if "O/C" not in self.table:
            self.table["O/C"] = self.table["O"] / self.table["C"]

        if "H/C" not in self.table:
            self.table["H/C"] = self.table["H"] / self.table["C"]

    def draw_density(
        self, 
        cmap:str ="Blues", 
        ax: plt.axes = None, 
        shade: bool = True
        ) -> None:
        """
        Draw Van-Krevelen density

        All parameters is optional

        Parameters
        ----------
        cmap: str
            color map
        ax: matplotlib ax
            external ax
        shade: bool
            show shade
        """
        sns.kdeplot(self.table["O/C"], self.table["H/C"], ax=ax, cmap=cmap, shade=shade)

    def draw_scatter(
        self, 
        ax:plt.axes = None, 
        volumes:float = None,
        color:str = 'blue',
        nitrogen:bool = False,
        sulphur:bool = False,
        alpha:float = 0.3, 
        mark_elem:str = None, 
        **kwargs) -> None:
        """
        plot Van-Krevelen diagramm

        All parameters is optional.

        Parameters
        ----------
        ax: Matplotlyb axes object
            send here ax if you want plot special graph
        volumes: float
            size of dot at diagram.
            By default calc by median intensity of spectrum
        color: str
            color of VK. Default blue
        nitrogen: bool
            mark nitrogen in brutto-formulas as orange
        sulphur: bool
            mark sulphur in brutto-formulas as green for CHOS and red for CHONS
        alpha: float
            transparency of dot at the scatter from 0 to 1
        mark_elem: str
            mark element in brutto-formulas by pink color
        **kwargs: dict
            dict for additional condition to scatter method        
        """
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        else:
            ax=ax
        
        if volumes is None:
            self.table['volume'] = self.table['intensity'] / self.table['intensity'].median()
        else:
            self.table['volume'] = volumes

        self.table['color'] = color

        if mark_elem is not None:
            
            self.table.loc[self.table[mark_elem] > 0, 'color'] = 'purple'

        if nitrogen and 'N' in self.table.columns:
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] > 0), 'color'] = 'orange'

        if sulphur and 'S' in self.table.columns:
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] < 1) & (self.table['S'] > 0), 'color'] = 'green'
            self.table.loc[(self.table['C'] > 0) & (self.table['H'] > 0) &(self.table['O'] > 0) & (self.table['N'] > 0) & (self.table['S'] > 0), 'color'] = 'red'
        
        if mark_elem is not None:
            ax.scatter(self.table.loc[self.table[mark_elem] > 0, 'O/C'], 
                        self.table.loc[self.table[mark_elem] > 0, 'H/C'],
                        s=self.table.loc[self.table[mark_elem] > 0, 'volume'], 
                        c='red', 
                        alpha=alpha, 
                        **kwargs)
        else:
            ax.scatter(self.table["O/C"], self.table["H/C"], s=self.table['volume'], c=self.table['color'], alpha=alpha, **kwargs)
        ax.set_xlabel("O/C")
        ax.set_ylabel("H/C")
        ax.yaxis.set_ticks(np.arange(0, 2.2, 0.4))
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.2)
        
        num_formules = self.table['C'].count()
        ax.set_title(f'{num_formules} formulas', size=10)

    def _plot_heatmap(self, df:pd.DataFrame) -> None:
        """Plot density map for VK

        Parameters
        ----------
        df: pd.DataFrame
            dataframe with density        
        """

        fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
        sns.heatmap(df.round(4),cmap='coolwarm',annot=True, linewidths=.5, ax=ax)
        bottom, top = ax.get_ylim()
        plt.yticks(rotation=0)
        plt.xticks(rotation=90) 
        ax.set_ylim(bottom + 0.5, top - 0.5)

        ax.set_xlabel('O/C')
        ax.set_ylabel('H/C')

    def squares(self, draw:bool = True) -> pd.DataFrame:
        """
        Calculate density  in VK divided into 20 squares

        Parameters
        ----------
        draw: bool
            Optional, default True. Draw heatmap for squares

        Return
        ------
        Pandas Dataframe with calculated square density
        """

        d_table = []
        sq = []
        table = copy.deepcopy(self.table)
        total_i = len(table)
        for y in [ (1.8, 2.2), (1.4, 1.8), (1, 1.4), (0.6, 1), (0, 0.6)]:
            hc = []
            for x in  [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]:
                temp = copy.deepcopy(self)
                temp.table = temp.table.loc[(temp.table['O/C'] >= x[0]) & (temp.table['O/C'] < x[1]) & (temp.table['H/C'] >= y[0]) & (temp.table['H/C'] < y[1])]
                temp_i = len(temp.table)
                hc.append(temp_i/total_i)
                sq.append(temp_i/total_i)
            d_table.append(hc)
        out = pd.DataFrame(data = d_table, columns=['0-0.25', '0,25-0.5','0.5-0.75','0.75-1'], index=['1.8-2.2', '1.4-1.8', '1-1.4', '0.6-1', '0-0.6'])
        self._plot_heatmap(out)

        # just for proper naming of squars. bad solution
        square = pd.DataFrame(data=sq, columns=['value'], index=[5,10,15,20,   4,9,14,19,   3,8,13,18,    2,7,12,17,   1,6,11,16])

        return square.sort_index()

    def scatter_density(self, ax=None, ax_x=None, ax_y=None, color:str='blue', alpha:float=0.3, volumes:float=None) -> None:
        """
        Plot VK scatter with density
        Same as joinplot in seaborn
        but you can use external axes

        Parameters
        ----------
        ax: matplotlib ax
            central ax for scatter
        ax_x: matplotlib ax
            horizontal ax for density
        ax_y: matplotlib ax
            vertical ax for density
        color: str
            color for scatter and density
        alpha: float
            alpha for scatter
        """
        if ax is None:
            fig = plt.figure(figsize=(6,6), dpi=100)
            gs = GridSpec(4, 4)

            ax = fig.add_subplot(gs[1:4, 0:3])
            ax_x = fig.add_subplot(gs[0,0:3])
            ax_y = fig.add_subplot(gs[1:4, 3])

        self.table['intensity'] = self.table['intensity'] / self.table['intensity'].median()

        if volumes is None:
            self.table['volume'] = self.table['intensity'] / self.table['intensity'].median()
        else:
            self.table['volume'] = volumes

        ax.scatter(self.table['O/C'],self.table['H/C'], s=self.table['volume'], alpha=alpha, c=color)
        ax.set_ylim(0,2.2)
        ax.set_xlim(0,1)
        ax.set_xlabel("O/C")
        ax.set_ylabel("H/C")
        ax.yaxis.set_ticks(np.arange(0, 2.2, 0.4))
        ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))

        oc = self.table['O/C']*self.table['intensity']
        hc = self.table['H/C']*self.table['intensity']
        total_int = self.table['intensity'].sum()

        x = np.linspace(self.table['O/C'].min(), self.table['O/C'].max(), 100)        
        oc = np.array([])
        for i, el in enumerate(x[1:]):
            s = self.table.loc[(self.table['O/C'] > x[i-1]) & (self.table['O/C'] <= el), 'intensity'].sum()
            coun = len(self.table) * s/total_int
            m = (x[i-1] + x[i])/2
            oc = np.append(oc, [m]*int(coun))
        sns.kdeplot(x = oc, ax=ax_x, color=color, fill=True, alpha=0.1, bw_adjust=2)
        ax_x.set_axis_off()
        ax_x.set_xlim(0,1)
        
        y = np.linspace(self.table['H/C'].min(), self.table['H/C'].max(), 100)        
        hc = np.array([])
        for i, el in enumerate(y[1:]):
            s = self.table.loc[(self.table['H/C'] > y[i-1]) & (self.table['H/C'] <= el), 'intensity'].sum()
            coun = len(self.table) * s/total_int
            m = (y[i-1] + y[i])/2
            hc = np.append(hc, [m]*int(coun))
        sns.kdeplot(x = hc, ax=ax_y, color=color, vertical=True, fill=True, alpha=0.1, bw_adjust=2)
        ax_y.set_axis_off()
        ax_y.set_ylim(0,2.2)


class ErrorTable(object):
    """
    A class used to recallibrate mass spectrum

    Attributes
    ----------
    table : pandas Datarame
        consist error table: error in ppm for mass
    """

    def __init__(
            self,
            table: pd.DataFrame = None,
    ) -> None:
        """
        Init ErrorTable object

        Parameters
        ----------
        table : pandas Datarame
            consist error table: error in ppm for mass
        """
        self.table = table

    def dif_mass(self) -> list:
        '''Generate common mass diffrence list

        Return:
        -------
        List of float, containing most common mass difference
        '''
        H = 1.007825
        C = 12.000000
        O = 15.994915

        dif = []
        for k in range(1,11):
            dif.append(k*(C + H*2))
            dif.append(k*(O))
            dif.append(k*(C + O))
            dif.append(k*(H*2))
            dif.append(k*(C*2 + O + H*2))
            dif.append(k*(C + O + H*2))
            dif.append(k*(O + H*2))
            dif.append(k*(C + O*2))

        return dif

    def md_error_map(
        self, 
        spec: "MassSpectrum", 
        ppm: float = 5, 
        show_map: bool = True
        ) -> pd.DataFrame:
        '''
        Calculate mass differnce map

        Parameters
        ----------
        spec: pd.Dataframe
            Dataframe with spectrum table from MassSpectrum
        ppm: float
            Optional. Default 5.
            Permissible error in ppm
        show_map: bool
            Optional. Default True.
            Show error in ppm versus mass

        Return
        ------
        Pandas Dataframe object with calculated error map
        '''

        dif = self.dif_mass()

        data = copy.deepcopy(spec.table)
        masses = data['mass'].values

        data = data.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        if len(data) > 1000:
            data = data[:1000]
        data = data.sort_values(by='mass').reset_index(drop=True)

        data_error = [] #array for new data

        for index, row in tqdm(data.iterrows(), total=len(data)): #take every mass in list
            
            mass = row["mass"]

            for i in dif:
                mz = mass + i #massdif

                idx = np.searchsorted(masses, mz, side='left')                
                if idx > 0 and (idx == len(masses) or np.fabs(mz - masses[idx - 1]) < np.fabs(mz - masses[idx])):
                    idx -= 1

                if np.fabs(masses[idx] - mz) / mz * 1e6 <= ppm:
                    data_error.append([mass, (masses[idx] - mz)/mz*1000000])
        
        df_error = pd.DataFrame(data = data_error, columns=['mass', 'ppm' ])
        
        if show_map:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            ax.scatter(df_error['mass'], df_error['ppm'], s=0.01)

        return df_error
    
    def fit_kernel(
        self, 
        f: np.array, 
        show_map: bool = True) -> pd.DataFrame:
        '''
        Fit max intesity of kernel density map

        Parameters
        ----------
        f: np.array
            keerndel density map in numpy array 100*100
        show_map: bool
            Optional. Default true.
            Plot how fit kernel

        Return
        ------
        Pandas Dataframe with error table for 100 values
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

    def kernel_density_map(
        self, 
        df_error: pd.DataFrame, 
        ppm: float = 3, 
        show_map: bool = False
        ) -> np.array:
        '''
        Plot kernel density map 100*100 for data

        Parameters
        ----------
        df_error: pd.Dataframe
            error_table for generate kerle density map
        ppm: float
            Optional. Default 3.
            treshould for generate
        show_map: bool
            Optional. Default True. plot kde

        Return
        ------
        numpy array 100*100 with generated kde
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

    def assign_error(
        self, 
        spec:MassSpectrum,
        ppm = 3,
        sign = '-',
        show_map:bool = True):
        '''
        Recallibrate by assign error

        Parameters
        ----------
        spec: MassSpectrum object
            Initial mass spectrum for recallibrate
        ppm: float
            Optional. Default 3.
            permissible relative error in callibrate error
        sign: str
            Optional. Default '-'. 
            for correct recallibration we need to mark mode
            '-' for negative
            '+' for positive
        show_error: bool
            Optional. Default True. Show process 

        Return
        ------
        ErrorTable object that contain recallabrate error ppm for mass diaposone

        '''
        spectr = copy.deepcopy(spec)
        spectr = spectr.assign(rel_error=ppm) 
        spectr = spectr.calculate_mass()
        spectr = spectr.calculate_error(sign=sign)
        spectr.show_error()

        error_table = spectr.table
        error_table = error_table.loc[:,['mass','rel_error']]
        error_table.columns = ['mass', 'ppm']
        error_table = error_table.dropna()

        kde = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kde, show_map=show_map)

        err['ppm'] = - err['ppm']
        err['mass'] = np.linspace(error_table['mass'].min(), error_table['mass'].max(),len(err))

        return ErrorTable(err)

    def massdiff_error(
        self,
        spec:MassSpectrum,
        show_map:bool = True):
        '''
        Self-recallibration of mass-spectra by mass-difference map

        Parameters
        -----------
        spec: MassSpectrum object
            Initial mass spectrum for recallibrate
        show_error: bool
            Optional. Default True. Show process 

        Return
        -------
        ErrorTable object that contain recallabrate error ppm for mass diaposone

        Reference
        ---------
        Smirnov, K. S., Forcisi, S., Moritz, F., Lucio, M., & Schmitt-Kopplin, P. 
        (2019). Mass difference maps and their application for the 
        recalibration of mass spectrometric data in nontargeted metabolomics. 
        Analytical chemistry, 91(5), 3350-3358. 
        '''
        spec_table = copy.deepcopy(spec)
        mde = self.md_error_map(spec = spec_table, show_map=show_map)
        f = self.kernel_density_map(df_error=mde)
        err = self.fit_kernel(f=f, show_map=show_map)
        err['mass'] = np.linspace(spec.table['mass'].min(), spec.table['mass'].max(),len(err))

        return ErrorTable(err)

    def etalon_error( self,
                    spec: "MassSpectrum", #initial masspectr
                    etalon: "MassSpectrum", #etalon massspectr
                    quart: float = 0.9, #treshold by quartile
                    ppm: float = 3,#treshold by ppm
                    show_error: bool = True
                    ): 
        '''
        Recallibrate by etalon

        Parameters
        ----------
        spec: MassSpectrum object
            Initial mass spectrum for recallibrate
        etalon: MassSpectrum object
            Etalon mass spectrum
        quart: float
            Optionaly. by default it is 0.9. 
            Usualy it is enough for good callibration
            Quartile, which will be taken for calc recallibrate error
        ppm: float
            Optionaly. Default 3.
            permissible relative error in ppm for seak peak in etalon
        show_error: bool
            Optional. Default True. Show process 

        Return
        ------
        ErrorTable object that contain recallabrate error ppm for mass diaposone

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

        return ErrorTable(err)

    def extrapolate(self, ranges:Tuple[float, float] = None) -> "ErrorTable":
        """
        Extrapolate error data

        Parameters
        ----------
        ranges: Tuple(numeric, numeric)
            Optionaly. Default None - all width of mass in error table.
            For which diaposone of mass extrapolate existin data

        Return
        ------
        ErrorTable object with extrapolated data
        """
        
        if ranges is None:
            ranges = [self.table['mass'].min(), self.table['mass'].max()]

        interpolation_range = np.linspace(ranges[0], ranges[1], 100)
        linear_interp = interp1d(self.table['mass'], self.table['ppm'],  bounds_error=False, fill_value='extrapolate')
        linear_results = linear_interp(interpolation_range)
        err = pd.DataFrame()
        err ['mass'] = interpolation_range
        err ['ppm'] = linear_results

        return ErrorTable(err)

    def show_error(self) -> None:
        """
        Plot error map from ErrorTable data
        """
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)
        ax.plot(self.table['mass'], self.table['ppm'])
        ax.set_xlabel('m/z, Da')
        ax.set_ylabel('error, ppm')

    def zeroshift(self, spec:"MassSpectrum") -> "ErrorTable":
        """
        Shift error so mean eror will be zero

        Parameters
        ----------
        spec: MassSpectrum object
            income massspec

        Return
        ------
        ErrorTable object with shifted ppm error
        """
        err = copy.deepcopy(self)
        mean_error = spec.drop_unassigned().calculate_error()['rel_error'].mean()
        err.table['ppm'] = err.table['ppm'] - mean_error
        return ErrorTable(err.table)       


class MassSpectrumList(object):
    """
    Class for work list of MassSpectrums objects
    
    Attributes
    ----------
    spectra: Sequence[MassSpectrum]
        list of MassSpectrum objects
    names: Optional[Sequence[str]]
        list of names for spectra
    """

    def __init__(self, spectra: Sequence[MassSpectrum] = None, names: Optional[Sequence[str]] = None):
        """
        init MassSpectrumList Class
        
        Parameters
        ----------
        spectra: Sequence[MassSpectrum]
            list of MassSpectrum objects
        names: Optional[Sequence[str]]
            list of names for spectra
        """
        
        if spectra:
            self.spectra = spectra
        else:
            self.spectra = []

        if names:
            self.names = names
        elif len(self.spectra) > 0:
            self.names = list(range(len(self.spectra)))
        else:
            self.names = []

    def calculate_similarity(self, mode: str = "cosine") -> np.ndarray:
        """
        Calculate similarity matrix for all spectra in MassSpectrumList

        Parameters
        ----------
        mode: str
            Optionaly. Default cosine. 
            one of the similarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"

        Return
        ------
        similarity matrix, 2d np.ndarray with size [len(names), len(names)]"""

        def get_vector(a, b):
            # FIXME Probably bad solution
            c = np.union1d(a, b)
            A = np.zeros(len(c), dtype=bool)
            B = np.zeros(len(c), dtype=bool)
            for i, el in enumerate(c):
                if el in a:
                    A[i] = True
                if el in b:
                    B[i] = True
            return A, B

        def jaccard(a, b):
            A, B = get_vector(a, b)
            return 1 - spatial.distance.jaccard(A, B)

        def tanimoto(a, b):
            A, B = get_vector(a, b)
            return 1 - spatial.distance.rogerstanimoto(A, B)

        def cosine(a, b):
            A, B = get_vector(a, b)
            return 1 - spatial.distance.cosine(A, B)

        if mode == "jaccard":
            similarity_func = jaccard
        elif mode == "tanimoto":
            similarity_func = tanimoto
        elif mode == 'cosine':
            similarity_func = cosine
        else:
            raise Exception(f"There is no such mode: {mode}")

        values = []
        for i in self.spectra:
            if 'calculated_mass' not in i.table:
                i = i.calculate_mass()
            values.append([])
            for j in self.spectra:
                if 'calculated_mass' not in j.table:
                    j = j.calculate_mass()
                values[-1].append(similarity_func(i.table['calculated_mass'].dropna().values, j.table['calculated_mass'].dropna().values))

        return np.array(values)

    def draw_similarity(
        self,
        mode: str = "cosine",
        values: np.ndarray = None,
        ax: plt.axes = None,
        annot = True
        ) -> None:
        """
        Draw similarity matrix by using seaborn

        Parameters
        ----------
        values: np.ndarray
            Optionaly. Similarity matix.
            Default None - It is call calculate_similarity() method.
        mode: str
            Optionaly. If values is none for calculate matrix. 
            Default cosine. one of the similarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"
        ax: matplotlib axes
            Entarnal axes for plot
        annotate: bool
            Draw value of similarity onto titles
        """
        if values is None:
            values = self.calculate_similarity(mode=mode)

        if ax is None:
            fig, ax = plt.subplots(figsize=(len(self.spectra),len(self.spectra)), dpi=75)
        
        x_axis_labels = self.names
        y_axis_labels = self.names
        sns.heatmap(np.array(values), vmin=0, vmax=1, cmap="viridis", annot=annot, ax=ax, xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        plt.title(mode)


class Tmds(object):
    """
    A class for calculate TMDS spectrum

    Attributes
    ----------
    table: pandas Datarame
        tmds spectrum - mass_dif, probability and caclulatedd parameters
    elems: Sequence[str]
        elements in brutto formulas
    """

    def __init__(
        self,
        table: pd.DataFrame = None,
        elems: Sequence[str] = None
        ) -> None:
        """
        Parameters
        ----------
        table: pandas Datarame
            Optional. tmds spectrum - mass_dif, probability and caclulatedd parameters
        elems: Sequence[str]
            Optional. elements in brutto formulas
        """

        self.elems = elems

        if table is None:
            self.table = pd.DataFrame()
        else:
            self.table = table
            self.elems = self.find_elems()

    def calc(
        self,
        mass_spec:"MassSpectrum",
        other:"MassSpectrum"=None,
        p: float = 0.2,
        wide: int = 10,
        C13_filter:bool = True,
        ) -> "Tmds":

        """
        Total mass difference statistic calculation 

        Parameters
        ----------
        mass_spec: MassSpectrum object
            for tmds calculation
        other: MassSpectrum object
            Optional. If None, TMDS will call by self.
        p: float
            Optional. Default 0.2. 
            Minimum relative probability for taking mass-difference
        wide: int
            Optional. Default 10.
            Minimum interval in 0.001*wide Da of peaeks.
        C13_filter: bool
            Optional. Default True. 
            Use only peaks that have C13 isotope peak

        Reference
        ---------
        Anal. Chem. 2009, 81, 10106
        """

        spec = copy.deepcopy(mass_spec)
        if other is None:
            spec2 = copy.deepcopy(mass_spec)
        else:
            spec2 = copy.deepcopy(other)

        if C13_filter:
            spec = spec.filter_by_C13(remove=True)
            spec2 = spec2.filter_by_C13(remove=True)
        else:
            spec = spec.drop_unassigned()
            spec2 = spec2.drop_unassigned()

        masses = spec.table['mass'].values
        masses2 = spec2.table['mass'].values

        mass_num = len(masses)
        mass_num2 = len(masses2)

        if mass_num <2 or mass_num2 < 2:
            raise Exception(f"Too low amount of assigned peaks")

        mdiff = np.zeros((mass_num, mass_num2), dtype=float)
        for x in range(mass_num):
            for y in range(x, mass_num2):
                dif = np.fabs(masses[x]-masses2[y])
                if dif < 300:
                    mdiff[x,y] = dif

        mdiff = np.round(mdiff, 3)
        unique, counts = np.unique(mdiff, return_counts=True)
        counts[0] = 0

        tmds_spec = pd.DataFrame()
        tmds_spec['mass_dif'] = unique
        tmds_spec['count'] = counts
        tmds_spec['probability'] = tmds_spec['count']/mass_num
        tmds_spec = tmds_spec.sort_values(by='mass_dif').reset_index(drop=True)

        value_zero = set([i/1000 for i in range (0, 300000)]) - set (unique)
        unique = np.append(unique, np.array(list(value_zero)))
        counts = np.append(counts, np.zeros(len(value_zero), dtype=float))

        peaks, properties = find_peaks(tmds_spec['probability'], distance=wide, prominence=p/2)
        prob = []
        for peak in peaks:
            prob.append(tmds_spec.loc[peak-5:peak+5,'probability'].sum())
        tmds_spec = tmds_spec.loc[peaks].reset_index(drop=True)
        tmds_spec['probability'] = prob
        tmds_spec = tmds_spec.loc[tmds_spec['probability'] > p]

        if len(tmds_spec) < 0:
            raise Exception(f"There isn't mass diff mass, decrease p-value")

        return Tmds(tmds_spec)
    
    def calc_by_brutto(
        self,
        mass_spec:"MassSpectrum"
        ) -> "Tmds":

        """
        Calculate self difference by calculated mass from brutto

        Parameters
        ----------
        mass_spec: MassSpectrum object
            for tmds calculation

        Return
        ------
        Tmds object with assigned signals and elements
        """

        mass = mass_spec.drop_unassigned().calculate_error().table['calculated_mass'].values
        massl = len(mass)
        mdiff = np.zeros((massl, massl), dtype=float)
        for x in range(massl):
            for y in range(x, massl):
                mdiff[x,y] = np.fabs(mass[x]-mass[y])

        mdiff = np.round(mdiff, 6)
        unique, counts = np.unique(mdiff, return_counts=True)
        counts[0] = 0

        diff_spec = pd.DataFrame()
        diff_spec['mass_dif'] = unique
        diff_spec['count'] = counts
        diff_spec['probability'] = diff_spec['count']/massl
        diff_spec = diff_spec.sort_values(by='mass_dif').reset_index(drop=True)

        return Tmds(diff_spec)

    def assign(
        self,
        generated_bruttos_table: pd.DataFrame = None,
        error: float = 0.001,
        gdf:dict = {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)},
        max_num: int = None
        ) -> "Tmds":

        """
        Finding the nearest mass in generated_bruttos_table

        Parameters
        ----------
        generated_bruttos_table: pandas DataFrame 
            Optional. with column 'mass' and elements, should be sorted by 'mass'
        error: float
            Optional. Default 0.001. 
            absolute error iin Da for assign formulas
        gdf: dict
            Optional, default {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}
            generate brutto table if generated_bruttos_table is None.
        max_num: int
            Optional. Default 100
        
        Return
        ------
        Tmds object with assigned signals and elements
        """

        if generated_bruttos_table is None:
            generated_bruttos_table = brutto_gen(gdf, rules=False)
            generated_bruttos_table = generated_bruttos_table.loc[generated_bruttos_table['mass'] > 0]

        table = self.table.copy()

        masses = generated_bruttos_table["mass"].values
        
        elems = list(generated_bruttos_table.drop(columns=["mass"]))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = []
        for index, row in table.iterrows():
            mass = row["mass_dif"]
            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1

            if np.fabs(masses[idx] - mass)  <= error:
                res.append({**dict(zip(elems, bruttos[idx])), "assign": True, "mass_dif":mass, "probability":row["probability"], "count":row["count"]})

        res = pd.DataFrame(res)

        if max_num is not None and len(res) > max_num:
            res = res.sort_values(by='count', ascending=False).reset_index(drop=True)
            res = res.loc[:max_num].reset_index(drop=True)
            res = res.sort_values(by='mass_dif').reset_index(drop=True)
        
        return Tmds(table=res, elems=elems)

    def find_elems(self):
        """
        Find elems from mass spectrum table.

        Find elements in table columns. Used elems_mass_table with all elements and isotopes.
        For example, element 'C' will be recognised as carbon 12C, element 'C_13" as 13C

        Returns
        -------
        list
            a list of found elemets. For example: ['C','H','O','N']
        """

        main_elems = elements_table()['element'].values
        all_elems = elements_table()['element_isotop'].values

        elems = []
        for col in self.table.columns:
            if col in main_elems:
                elems.append(col)
            elif col in all_elems:
                elems.append(col)

        return elems

    def calculate_mass(self) -> "Tmds":
        """
        Calculate mass from brutto formulas in tmds table

        Return
        ------
        Tmds object with calculated mass for assigned brutto formulas
        """
        
        table = copy.deepcopy(self.table)
        self.elems = self.find_elems()
        
        table = table.loc[:,self.elems]

        masses = get_elements_masses(self.elems)

        self.table["calculated_mass"] = table.multiply(masses).sum(axis=1)
        self.table["calculated_mass"] = np.round(self.table["calculated_mass"], 6)
        self.table.loc[self.table["calculated_mass"] == 0, "calculated_mass"] = np.NaN

        return Tmds(self.table, elems=self.elems)

    def draw(
        self,
        xlim: Tuple[float, float] = (None, None),
        ylim: Tuple[float, float] = (None, None),
        color: str = 'black',
        ax = None,
        ) -> None:
        """
        Draw TMDS spectrum

        All parameters is optional

        Parameters
        ----------
        xlim: Tuple (float, float)
            restrict for mass
        ylim: Tuple (float, float)
            restrict for probability
        color: str
            color of draw
        ax: matplotlyp axes object
            send here ax to plot in your own condition
        """

        df = self.table.sort_values(by="mass_dif")

        mass = df['mass_dif'].values
        if xlim[0] is None:
            xlim = (mass.min(), xlim[1])
        if xlim[1] is None:
            xlim = (xlim[0], mass.max())

        intensity = df['probability'].values
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
            fig, ax = plt.subplots(figsize=(4,4), dpi=75)
            
        ax.plot(M, I, color=color, linewidth=0.2)
        ax.plot([xlim[0], xlim[1]], [0, 0], color=color, linewidth=0.2)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('mass difference, Da')
        ax.set_ylabel('P')
        ax.set_title(f'{len(self.table)} peaks')
        return

    def save(self, filename:str) -> None:
        """
        Save Tmds spectrum as csv

        Parameters
        ----------
        filename: str
            file name with path in which save tmds
        """
        self.table.to_csv(filename)

    def load(self, filename:str) -> "Tmds":
        """
        Load Tmds spectrum table from csv

        Parameters
        ----------
        filename: str
            file name with path in which load tmds
        """
        return Tmds(pd.read_csv(filename))


if __name__ == '__main__':
    pass