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

from heapq import merge
from pathlib import Path
from typing import List, Dict, Sequence, Union, Optional, Mapping, Tuple
import copy
from collections import UserDict, UserList

import matplotlib.pyplot as plt
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
from .metadata import MetaData


class MassSpectrum(object):
    """ 
    A class used to represent mass spectrum

    Attributes
    ----------
    table : pandas Datarame
        Optional. consist spectrum (mass and intensity of peaks) and all calculated parameters
        like brutto formulas, calculated mass, relative errorr
    metadata: MetaData
        Optional. Default None. Metadata object that consist dictonary of metadata
    """

    def __init__(
                self,
                table: Optional[pd.DataFrame] = None,
                metadata: Optional[Dict] = None
                ) -> pd.DataFrame:
        """
        Parameters
        ----------
        table : pandas Datarame
            Optional. Consist spectrum (mass and intensity of peaks) and all calculated 
            parameters like brutto formulas, calculated mass, relative errorr
        metadata: Dict
            Optional. Default None. To add some data into spectrum metedata. 
        """

        self.features = ["mass", 'intensity', "calculated_mass", 'intensity', "rel_error"]

        if table is not None:
            self.table = table
        else:
            self.table = pd.DataFrame(columns=["mass", 'intensity', "brutto", "calculated_mass", "rel_error"])

        self.metadata = MetaData(metadata)

    def _copy(func):
        """
        Decorator for deep copy self before apllying methods
        
        Parameters
        ----------
        func: method
            function for decoarate
        
        Return
        ------
        function with deepcopyed self

        """
        def wrapped(self, *args, **kwargs):
            self = copy.deepcopy(self)
            args = tuple([copy.deepcopy(arg) if isinstance(arg, MassSpectrum) else arg for arg in args])
            kwargs = {k: copy.deepcopy(v) if isinstance(v, MassSpectrum) else v for k, v in kwargs.items()}
            return func(self, *args, **kwargs)
        return wrapped

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
        assign_mark: bool = False,
        metadata: Optional[Dict] = {}
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
        assign_mark: bool
            default False. Mark peaks as assigned if they have elements
            need for load mass-list treated by external software
        metadata: Dict
            Optional. Default None. Metadata object that consist dictonary of metadata.
            if name not in metadata - name will take from filename.

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

        if assign_mark:
            self._mark_assigned_by_brutto()

        self.table = self.table.sort_values(by="mass").reset_index(drop=True)

        if 'name' not in metadata:
            self.metadata['name'] = filename.split('/')[-1].split('.')[0]

        self.metadata.add(metadata=metadata)

        return self

    def _mark_assigned_by_brutto(self) -> None:
        """Mark paeks in loaded mass list if they have brutto

        Return
        ------
        MassSpectrum object with assigned mark
        """

        assign = []
        elems = self.find_elems()
        for i, row in self.table.iterrows():
            flag = False
            for el in elems:
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
            brutto_dict: dict = None,
            generated_bruttos_table: pd.DataFrame = None,
            rel_error: float = None,
            abs_error: float = None,
            sign: str ='-',
            mass_min: float =  None,
            mass_max: float = None,
    ) -> "MassSpectrum":
        """
        Finding the nearest mass in generated_bruttos_table
        
        Parameters
        -----------
        brutto_dict: dict
            Optional. Deafault None.
            Custom Dictonary for generate brutto table.
            Example: {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'C_13':(0,3)}
        generated_bruttos_table: pandas DataFrame 
            Optional. Contain column 'mass' and elements, 
            should be sorted by 'mass'.
            Can be generated by function brutto_generator.brutto_gen(). 
            if 'None' generate table with default elemnets and ranges
            C: 4-50, H 4-100, O 0-25, N 0-3, S 0-2.
        rel_error: float
            Optional. default 0.5, permissible error in ppm for assign mass to brutto formulas
        abs_error: float
            Optional. default None, permissible absolute error for assign mass to brutto formulas
        sign: str
            Optional. Deafult '-'.
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            '0' for neutral
        mass_min: float
            Optional. Default None. Minimall mass for assigment
        mass_max: float
            Optional. Default None. Maximum mass for assigment   

        Return
        ------
        MassSpectra object with assigned signals
        """

        if generated_bruttos_table is None:
            generated_bruttos_table = brutto_gen(brutto_dict)

        if mass_min is None:
            mass_min = self.table['mass'].min()
        if mass_max is None:
            mass_max = self.table['mass'].max()

        self.table = self.table.loc[:,['mass', 'intensity']]
        table = self.table.loc[(self.table['mass']>=mass_min) & (self.table['mass']<=mass_max)].copy()

        masses = generated_bruttos_table["mass"].values
        
        if sign == '-':
            mass_shift = - 0.00054858 + 1.007825  # electron and hydrogen mass
        elif sign == '+':
            mass_shift = 0.00054858  # electron mass
        elif sign == '0':
            mass_shift = 0
        else:
            raise Exception('Sended sign to assign method is not correct. May be "+","-","0"')

        self.metadata.add({'sign':sign})

        if rel_error is not None:
            rel = True
        if abs_error is not None:
            rel = False
        if rel_error is not None and abs_error is not None:
            raise Exception('one of rel_error or abs_error must be None in assign method')
        if rel_error is None and abs_error is None:
            rel = True
            rel_error = 0.5

        elems = list(generated_bruttos_table.drop(columns=["mass"]))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = []
        for index, row in table.iterrows():
            mass = row["mass"] + mass_shift
            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1

            if rel:
                if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error:
                    res.append({**dict(zip(elems, bruttos[idx])), "assign": True})
                else:
                    res.append({"assign": False})
            else:
                if np.fabs(masses[idx] - mass) <= abs_error:
                    res.append({**dict(zip(elems, bruttos[idx])), "assign": True})
                else:
                    res.append({"assign": False})

        res = pd.DataFrame(res)

        table = table.join(res)
        self.table = self.table.merge(table, how='outer', on=list(self.table.columns))
        self.table['assign'] = self.table['assign'].fillna(False)

        return self

    @_copy
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
        
        self.table = self.table.sort_values(by='mass').reset_index(drop=True)
        
        flags = np.zeros(self.table.shape[0], dtype=bool)
        masses = self.table["mass"].values
        
        C13_C12 = 1.003355  # C13 - C12 mass difference

        
        for index, row in self.table.iterrows():
            mass = row["mass"] + C13_C12
            error = mass * rel_error * 0.000001

            idx = np.searchsorted(masses, mass, side='left')
            
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1
            
            if np.fabs(masses[idx] - mass)  <= error:
                flags[index] = True
        
        self.table['C13_peak'] = flags

        if remove:
            self.table = self.table.loc[(self.table['C13_peak'] == True) & (self.table['assign'] == True)].reset_index(drop=True)

        return self

    @_copy
    def calculate_brutto(self) -> 'MassSpectrum':
        """
        Calculate brutto formulas from assign table

        Return
        ------
        MassSpectrum object wit calculated bruttos
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        elems = self.find_elems()
        out = []
        for i, row in self.table.iterrows():
            s = ''
            for el in elems:
                if len(el.split('_')) == 2:
                    ele = f'({el})'
                else:
                    ele = el
                if row[el] == 1:
                    s = s + f'{ele}'
                elif row[el] > 0:
                    s = s + f'{ele}{int(row[el])}'
            out.append(s)
        
        self.table['brutto'] = out

        return self

    def copy(self) -> 'MassSpectrum':
        """
        Deepcopy of self MassSpectrum object

        Return
        ------
        Deepcopy of self MassSpectrum object
        """
        return copy.deepcopy(MassSpectrum(self.table))

    @_copy
    def _calculate_sign(self) -> str:
        """
        Calculate sign from mass and calculated mass

        Return
        ------
        str: "-", "+", "0"
        """
        self = self.drop_unassigned()

        value = (self.table["calculated_mass"]- self.table["mass"]).mean()
        value = np.round(value,4)
        if value > 1:
            return '-'
        elif value > 0.0004 and value < 0.01:
            return '+'
        else:
            return '0'

    @_copy
    def calculate_error(self, sign:str=None) -> "MassSpectrum":
        """
        Calculate relative and absolute error of assigned peaks

        Parameters
        ----------
        sign: str
            Optional. Default None and calculated by self. 
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            '0' for neutral
        
        Return
        ------
        MassSpectrum object wit calculated error
        """
        if "calculated_mass" not in self.table:
            self = self.calculate_mass()

        if sign is None:
            if 'sign' in self.metadata:
                sign = self.metadata['sign']
            else:
                sign = self._calculate_sign()

        if sign == '-':
            self.table["abs_error"] = self.table["mass"] - self.table["calculated_mass"] + (- 0.00054858 + 1.007825) #-electron + proton
        elif sign == '+':
            self.table["abs_error"] = self.table["mass"] - self.table["calculated_mass"] + 0.00054858 #+electron
        elif sign == '0':
            self.table["abs_error"] = self.table["mass"] - self.table["calculated_mass"]
        else:
            raise Exception('Sended sign or sign in metadata is not correct. May be "+","-","0"')
        
        self.table["rel_error"] = self.table["abs_error"] / self.table["mass"] * 1e6
        
        return self

    @_copy
    def calculate_mass(self) -> "MassSpectrum":
        """
        Calculate mass from assigned brutto formulas

        Return
        ------
        MassSpectrum object with calculated mass
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")
        
        elems = self.find_elems()
        
        table = self.table.loc[:,elems].copy()
        
        masses = get_elements_masses(elems)

        self.table["calculated_mass"] = table.multiply(masses).sum(axis=1)
        self.table["calculated_mass"] = np.round(self.table["calculated_mass"], 6)
        self.table.loc[self.table["calculated_mass"] == 0, "calculated_mass"] = np.NaN

        return self
    
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

    @_copy
    def __or__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        """
        Logic function or for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain all assigned brutto formulas from two spectrum
        """
        
        self = self.normalize()
        other = other.normalize()

        if "calculated_mass" not in self.table:
            self = self.calculate_mass()
        if "calculated_mass" not in other.table:
            other = other.calculate_mass()

        a = self.table.dropna()
        b = other.table.dropna()
        
        a = a.append(b, ignore_index=True)
        a = a.drop_duplicates(subset=['calculated_mass'])

        return MassSpectrum(a)

    @_copy
    def __xor__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        """
        Logic function xor for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain xor assigned brutto formulas from two spectrum
        """

        sub1 = self.__sub__(other)
        sub2 = other.__sub__(self)
        
        return sub1.__or__(sub2)

    @_copy
    def __and__(self: "MassSpectrum", other: "MassSpectrum") -> "MassSpectrum":
        """
        Logic function and for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain common assigned brutto formulas from two spectrum
        """

        self = self.normalize()
        other = other.normalize()

        if "calculated_mass" not in self.table:
            self = self.calculate_mass()
        if "calculated_mass" not in other.table:
            other = other.calculate_mass()

        a = self.table['calculated_mass'].dropna().to_list()
        b = other.table['calculated_mass'].dropna().to_list()
        
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

    @_copy
    def __sub__(self, other:"MassSpectrum") -> "MassSpectrum":
        """
        Logic function substraction for two MassSpectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        MassSpectrum object contain substraction assigned brutto formulas from two spectrum
        """
        
        self = self.normalize()
        other = other.normalize()

        if "calculated_mass" not in self.table:
            self = self.calculate_mass()
        if "calculated_mass" not in other.table:
            other = other.calculate_mass()

        a = self.table['calculated_mass'].dropna().to_list()
        b = other.table['calculated_mass'].dropna().to_list()
        
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

    @_copy
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
        self = self.normalize()
        other = other.normalize()

        if "calculated_mass" not in self.table:
            self = self.calculate_mass()
        if "calculated_mass" not in other.table:
            other = other.calculate_mass()

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

    @_copy
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

        self.table = self.table.loc[self.table["assign"] == True].reset_index(drop=True)

        return self

    @_copy
    def sum_isotopes(self) -> "MassSpectrum":
        """
        All isotopes will be sum and title as main.

        Return
        ------
        MassSpectrum object without minor isotopes        
        """

        elems = self.find_elems()
        for el in elems:
            res = el.split('_')
            if len(res) == 2:
                if res[0] not in self.table:
                    self.table[res[0]] = 0
                self.table[res[0]] = self.table[res[0]] + self.table[el]
                self.table = self.table.drop(columns=[el]) 

        return self

    @_copy
    def calculate_simmilarity(self, other:"MassSpectrum", mode:str='tanimoto') -> float:
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
        self = self.normalize()
        other = other.normalize()

        if 'calculated_mass' not in self.table:
            self = self.calculate_mass()
        if 'calculated_mass' not in other.table:
            other = other.calculate_mass()

        s1 = self.drop_unassigned().normalize(how='sum')
        s2 = other.drop_unassigned().normalize(how='sum')

        df1 = pd.DataFrame()
        df1['cmass'] = s1.drop_unassigned().table['calculated_mass']
        df1['intens'] = s1.drop_unassigned().table['intensity']

        df2 = pd.DataFrame()
        df2['cmass'] = s2.drop_unassigned().table['calculated_mass']
        df2['intens'] = s2.drop_unassigned().table['intensity']

        res = df1.merge(df2, how='outer', on='cmass')
        res.fillna(0, inplace=True)

        a = res['intens_x'].values
        b = res['intens_y'].values

        a = a/np.sum(a)
        b = b/np.sum(b)      

        if mode == "jaccard":
            m1 = set(df1['cmass'].to_list())
            m2 = set(df2['cmass'].to_list())
            return len(m1 & m2)/len(m1 | m2)
        elif mode == "tanimoto":
            return np.dot(a, b)/(np.dot(a, a) + np.dot(b, b) - np.dot(a, b))
        elif mode == 'cosine':
            return 1 - spatial.distance.cosine(a, b)
        else:
            raise Exception(f"There is no such mode: {mode}")

    @_copy
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
        if "DBE" not in self.table:
            self = self.calculate_dbe()        

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

        table = self.copy().sum_isotopes().table
        self.table['CRAM'] = table.apply(check, axis=1)

        return self

    @_copy
    def get_cram_value(self) -> int:
        """
        Calculate percent of CRAM molecules
        (carboxylic-rich alicyclic molecules)

        Return
        ------
        int. percent of CRAM molecules in mass-spec
        weight by intensity
        """
        if "CRAM" not in self.table:
            self = self.calculate_cram()

        value = self.table.loc[self.table['CRAM'] == True, 'intensity'].sum()/self.table.loc[self.table['assign']==True, 'intensity'].sum()
        return int(value*100)

    @_copy
    def calculate_ai(self) -> 'MassSpectrum':
        """
        Calculate AI

        Return
        ------
        MassSpectrum object with calculated AI
        """
        if "DBE_AI" not in self.table:
            self = self.calculate_dbe_ai()

        if "CAI" not in self.table:
            self = self.calculate_cai()

        self.table["AI"] = self.table["DBE_AI"] / self.table["CAI"]

        return self

    @_copy
    def calculate_cai(self) -> 'MassSpectrum':
        """
        Calculate CAI

        Return
        ------
        MassSpectrum object with calculated CAI
        """
        
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table

        for element in "CONSP":
            if element not in table:
                table[element] = 0

        self.table['CAI'] = table["C"] - table["O"] - table["N"] - table["S"] - table["P"]

        return self

    @_copy
    def calculate_dbe_ai(self) -> 'MassSpectrum':
        """
        Calculate DBE

        Return
        ------
        MassSpectrum object with calculated DBE
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table

        for element in "CHONPS":
            if element not in table:
                table[element] = 0

        self.table['DBE_AI'] = 1.0 + table["C"] - table["O"] - table["S"] - 0.5 * (table["H"] + table['N'] + table["P"])

        return self

    @_copy
    def calculate_dbe(self) -> 'MassSpectrum':
        """
        Calculate DBE

        Return
        ------
        MassSpectrum object with calculated DBE
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table

        for element in "CHON":
            if element not in table:
                table[element] = 0

        self.table['DBE'] = 1.0 + table["C"] - 0.5 * (table["H"] - table['N'])

        return self

    @_copy
    def calculate_dbe_o(self) -> 'MassSpectrum':
        """
        Calculate DBE-O

        Return
        ------
        MassSpectrum object with calculated DBE-O
        """
        if "DBE" not in self.table:
            self = self.calculate_dbe()

        table = self.sum_isotopes().table
        self.table['DBE-O'] = table['DBE'] - table['O']

        return self

    @_copy
    def calculate_dbe_oc(self) -> 'MassSpectrum':
        """
        Calculate DBE-O/C

        Return
        ------
        MassSpectrum object with calculated DBE-O/C
        """
        if "DBE" not in self.table:
            self = self.calculate_dbe()

        table = self.sum_isotopes().table
        self.table['DBE-OC'] = (table['DBE'] - table['O'])/table['C']

        return self

    @_copy
    def calculate_hc_oc(self) -> 'MassSpectrum':
        """
        Calculate H/C and O/C

        Return
        ------
        MassSpectrum object with calculated H/C O/C
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table
        self.table['H/C'] = table['H']/table['C']
        self.table['O/C'] = table['O']/table['C']

        return self

    @_copy
    def calculate_kendrick(self) -> 'MassSpectrum':
        """
        Calculate Kendrick mass and Kendrick mass defect

        Return
        ------
        MassSpectrum object with calculated Ke and KMD
        """

        if 'calculated_mass' not in self.table:
            self = self.calculate_mass()

        self.table['Ke'] = self.table['calculated_mass'] * 14/14.01565
        self.table['KMD'] = np.floor(self.table['calculated_mass'].values) - np.array(self.table['Ke'].values)
        self.table.loc[self.table['KMD']<=0, 'KMD'] = self.table.loc[self.table['KMD']<=0, 'KMD'] + 1

        return self

    @_copy
    def calculate_nosc(self) -> 'MassSpectrum':
        """
        Calculate Normal oxidation state of carbon (NOSC).

        Notes
        -----
        >0 - oxidate state.
        <0 - reduce state.
        0 - neutral state

        Return
        ------
        MassSpectrum object with calculated DBE

        Reference
        ---------
        Boye, Kristin, et al. "Thermodynamically 
        controlled preservation of organic carbon 
        in floodplains."
        Nature Geoscience 10.6 (2017): 415-419.
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table

        for element in "CHONS":
            if element not in table:
                table[element] = 0

        self.table['NOSC'] = 4.0 - (table["C"] * 4 + table["H"] - table['O'] * 2 - table['N'] * 3 - table['S'] * 2)/table['C']

        return self
    
    @_copy
    def calculate_mol_class_zones(self) -> "MassSpectrum":
        """
        Assign molecular class for formulas

        Return
        ------
        MassSpectrum object with assigned zones
        """

        if 'AI' not in self.table:
            self = self.calculate_ai()
        if 'H/C' not in self.table or 'O/C' not in self.table:
            self = self.calculate_hc_oc()

        table = self.sum_isotopes().table

        for element in "CHON":
            if element not in table:
                table[element] = 0

        def get_zone(row):

            if row['H/C'] >= 1.5:
                if row['O/C'] < 0.3 and row['N'] == 0:
                    return 'lipids'
                elif row['N'] >= 1:
                    return 'N-satureted'
                else:
                    return 'aliphatics'
            elif row['H/C'] < 1.5 and row['AI'] < 0.5:
                if row['O/C'] <= 0.5:
                    return 'unsat_lowOC'
                else:
                    return 'unsat_highOC'
            elif row['AI'] > 0.5 and row['AI'] <= 0.67:
                if row['O/C'] <= 0.5:
                    return 'aromatic_lowOC'
                else:
                    return 'aromatic_highOC'
            elif row['AI'] > 0.67:
                if row['O/C'] <= 0.5:
                    return 'condensed_lowOC'
                else:
                    return 'condensed_highOC'
            else:
                return 'undefinded'
                
        self.table['class'] = table.apply(get_zone, axis=1)

        return self

    @_copy
    def get_mol_class_density(self, weight: str = "intensity") -> dict:
        """
        get molercular classes

        Parameters
        ----------
        weight: str
            how calculate density. Default "intensity".
            Also can be "count".

        Return
        ------
        Dict. mol_class:density
        
        References
        ----------
        Zherebker, Alexander, et al. "Interlaboratory comparison of 
        humic substances compositional space as measured by Fourier 
        transform ion cyclotron resonance mass spectrometry 
        (IUPAC Technical Report)." 
        Pure and Applied Chemistry 92.9 (2020): 1447-1467.
        """

        ans = {}
        self = self.drop_unassigned().calculate_mol_class_zones()
        count_density = len(self.table)
        sum_density = self.table["intensity"].sum()

        for zone in ['unsat_lowOC',
                    'unsat_highOC',
                    'condensed_lowOC',
                    'condensed_highOC',
                    'aromatic_lowOC',
                    'aromatic_highOC',
                    'aliphatics',            
                    'lipids',
                    'N-satureted',
                    'undefinded']:

            if weight == "count":
                ans[zone] = len(self.table.loc[self.table['class'] == zone])/count_density

            elif weight == "intensity":
                ans[zone] = self.table.loc[self.table['class'] == zone, 'intensity'].sum()/sum_density

            else:
                raise ValueError(f"weight should be count or intensity not {weight}")
        
        return ans

    @_copy
    def calculate_all(self) -> "MassSpectrum":
        """
        Calculated all avaible in this lib metrics

        Return
        ------
        MassSpectrum object with calculated metrics
        """

        self = self.calculate_mass()
        self = self.calculate_error()
        self = self.calculate_dbe()
        self = self.calculate_dbe_o()
        self = self.calculate_ai()
        self = self.calculate_dbe_oc()
        self = self.calculate_dbe_ai()
        self = self.calculate_mol_class_zones()
        self = self.calculate_hc_oc()
        self = self.calculate_cai()
        self = self.calculate_cram()
        self = self.calculate_nosc()
        self = self.calculate_brutto()
        self = self.calculate_mol_class_zones()
        self = self.calculate_kendrick()

        return MassSpectrum(self.table)

    @_copy
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

        if how=='max':
            self.table['intensity'] /= self.table['intensity'].max()
        elif how=='sum':
            self.table['intensity'] /= self.table['intensity'].sum()
        elif how=='median':
            self.table['intensity'] /= self.table['intensity'].median()
        elif how=='mean':
            self.table['intensity'] /= self.table['intensity'].mean()
        else:
            raise Exception(f"There is no such mode: {how}")

        return self

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
    
    @_copy
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
        self.table = self.table.reset_index(drop=True)
        wide = len(err)

        min_mass = err['mass'].min()
        max_mass = err['mass'].max()
        a = np.linspace(min_mass, max_mass, wide+1)

        for i in range(wide):
            for ind in self.table.loc[(self.table['mass']>a[i]) & (self.table['mass']<a[i+1])].index:
                mass = self.table.loc[ind, 'mass']
                e = mass * err.loc[i, 'ppm'] / 1000000
                self.table.loc[ind, 'mass'] = self.table.loc[ind, 'mass'] + e
                
        return self

    @_copy
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
            Tmds object, include table with most intensity mass difference
        abs_error: float
            Optional, default 0.001. Error for assign peaks by massdif
        p: float
            Optional. Default 0.2. 
            Relative intensity coefficient for treshold tmds spectrum
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

        tmds = tmds_spec.table.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        tmds = tmds.loc[tmds['intensity'] > p]
        elem = tmds_spec.find_elems()

        assign_false = copy.deepcopy(self.table.loc[self.table['assign'] == False]).reset_index(drop=True)
        assign_true = copy.deepcopy(self.table.loc[self.table['assign'] == True]).reset_index(drop=True)
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

    @_copy
    def calculate_DBEvsO(self, 
                        olim: Optional[Tuple[int, int]] = None, 
                        draw: Optional[bool] = True, 
                        ax: Optional[plt.axes] = None, 
                        **kwargs: dict) -> Tuple[float, float]:
        """
        Draw plot DBE by nO and calculate linear fit
        
        Parameters
        ----------
        olim: tuple of two int
            limit for nO. Deafult None
        draw: bool
            draw scatter DBE vs nO and how it is fitted
        ax: matplotlib axes
            ax fo outer plot. Default None
        **kwargs: dict
            dict for additional condition to scatter matplotlib

        Return
        ------
        (float, float), a and b in fit y = a*x + b

        References
        ----------
        Bae, E., Yeo, I. J., Jeong, B., Shin, Y., Shin, K. H., & Kim, S. (2011). 
        Study of double bond equivalents and the numbers of carbon and oxygen 
        atom distribution of dissolved organic matter with negative-mode FT-ICR MS.
        Analytical chemistry, 83(11), 4193-4199.
        
        """
        if 'DBE' not in self.table:
            self = self.calculate_dbe()

        self = self.drop_unassigned()
        if olim is None:
            no = list(range(5, int(self.table['O'].max())-4))
        else:
            no = list(range(olim[0],olim[1]))

        dbe_o = []
        
        for i in no:
            dbes = self.table.loc[self.table['O'] == i, 'DBE']
            intens = self.table.loc[self.table['O'] == i, 'intensity']
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
        
        if draw:
            if ax is None:
                fig,ax = plt.subplots(figsize=(3,3), dpi=100)
            
            ax.scatter(x, y, **kwargs)
            ax.plot(x, linear(x, *popt), label=f'y={round(popt[0],2)}x + {round(popt[1],1)} R2={round(r_squared, 4)}', **kwargs)
            ax.set_xlim(4)
            ax.set_ylim(5)
            ax.set_xlabel('number of oxygen')
            ax.set_ylabel('DBE average')
            ax.legend()

        return popt[0], popt[1]

    def vk_squares(self, ax=None, draw:bool=True) -> pd.DataFrame:
        """
        Calculate density in Van Krevelen diagram divided into 20 squares

        Parameters
        ----------
        ax: matplotlib ax
            Optional. external ax
        draw: bool
            Optional. Default True. Plot heatmap

        Return
        ------
        Pandas Dataframe with calculated square density
        """

        if 'H/C' not in self.table or 'O/C' not in self.table:
            self = self.calculate_hc_oc()   

        d_table = []
        sq = []
        table = self.table
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

        if draw:
            if ax is None:
                fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
            sns.heatmap(out.round(4),cmap='coolwarm',annot=True, linewidths=.5, ax=ax)
            bottom, top = ax.get_ylim()
            plt.yticks(rotation=0)
            plt.xticks(rotation=90) 
            ax.set_ylim(bottom + 0.5, top - 0.5)

            ax.set_xlabel('O/C')
            ax.set_ylabel('H/C')

        # just for proper naming of squars. bad solution
        square = pd.DataFrame(data=sq, columns=['value'], index=[5,10,15,20,   4,9,14,19,   3,8,13,18,    2,7,12,17,   1,6,11,16])
        
        return square.sort_index()
        

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
        show_map: bool = False
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
            Optional. Default False.
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

        for index, row in data.iterrows(): #take every mass in list
            
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
        mass: np.array,
        err_ppm: float = 3,
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
        
        xmin = min(mass)
        xmax = max(mass)
        
        #FIXME constan 100 maybe not good idea
        kde_err['mass'] = np.linspace(xmin, xmax, 100)

        ymin = -err_ppm
        ymax = err_ppm

        if show_map:
            fig = plt.figure(figsize=(4,4), dpi=75)
            ax = fig.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.imshow(df, extent=[xmin, xmax, ymin, ymax], aspect='auto')
            ax.plot(kde_err['mass'], kde_err['ppm'], c='r')
            ax.set_xlabel('m/z, Da')
            ax.set_ylabel('error, ppm')      

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

        #FIXME constan 100 maybe not good idea
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
        show_error: bool
            Optional. Default False. Show process 

        Return
        ------
        ErrorTable object that contain recallabrate error ppm for mass diaposone

        '''
        spectr = copy.deepcopy(spec)
        spectr = spectr.assign(rel_error=ppm).calculate_mass().calculate_error()

        error_table = spectr.table
        error_table = error_table.loc[:,['mass','rel_error']]
        error_table.columns = ['mass', 'ppm']
        error_table['ppm'] = - error_table['ppm']
        error_table = error_table.dropna()

        kde = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kde, show_map=show_map, mass=spec.table['mass'].values)

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
        spec = copy.deepcopy(spec)
        mde = self.md_error_map(spec = spec)
        f = self.kernel_density_map(df_error=mde)
        err = self.fit_kernel(f=f, show_map=show_map, mass=spec.table['mass'].values)
        
        return ErrorTable(err)

    def etalon_error( self,
                    spec: "MassSpectrum", #initial masspectr
                    etalon: "MassSpectrum", #etalon massspectr
                    quart: float = 0.9, #treshold by quartile
                    ppm: float = 3,#treshold by ppm
                    show_map: bool = True
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
        show_map: bool
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
        df['ppm']=df['dif']/df['mass']*1000000

        error_table = df.loc[:,['mass','ppm']]
        error_table = error_table.dropna()

        kde = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kde, show_map=show_map, mass=spec.table['mass'].values)

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


class MassSpectrumList(UserList):
    """
    Class for work list of MassSpectrums objects
    inheritan from list class with some extra features.
    Store list of MassSpectrum objects
    """

    def __init__(self, spectra: Optional[List["MassSpectrum"]] = []):

        t = type(MassSpectrum())
        for spec in spectra:
            if isinstance(spec, t) == False:
                raise Exception(f'MassSpectrumList must contain only MassSpectrum objects, not {type(spec)}')

        super().__init__(spectra)
        """
        init MassSpectrumList Class
        
        Parameters
        ----------
        spectra: Sequence[MassSpectrum]
            list of MassSpectrum objects
        """

    def calculate_similarity(self, mode: str = "cosine", symmetric = True) -> np.ndarray:
        """
        Calculate similarity matrix for all spectra in MassSpectrumList

        Parameters
        ----------
        mode: str
            Optionaly. Default cosine. 
            one of the similarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"
        symmetric: bool
            Optionaly. Default True.
            If metric is symmtrical ( a(b)==b(a) ) it is enough to calc just half of table

        Return
        ------
        similarity matrix, 2d np.ndarray with size [len(names), len(names)]"""

        
        spec_num = len(self)
        values = np.eye(spec_num)

        for x in range(spec_num):
            if symmetric:
                for y in range(x+1, spec_num):
                    values[x,y] = self[x].calculate_simmilarity(self[y], mode=mode)
            else:
                for y in range(spec_num):
                    values[x,y] = self[x].calculate_simmilarity(self[y], mode=mode)
        
        if symmetric:
            values = values + values.T - np.diag(np.diag(values))

        return values

    def draw_similarity(
        self,
        mode: str = "cosine",
        values: np.ndarray = None,
        ax: plt.axes = None,
        annot = True,
        **kwargs
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
        **kwargs: dict
            Additional parameters to seaborn heatmap method
        """
        if values is None:
            values = self.calculate_similarity(mode=mode)

        if ax is None:
            fig, ax = plt.subplots(figsize=(len(self),len(self)), dpi=75)
        
        axis_labels = []
        for i, spec in enumerate(self):
            axis_labels.append(spec.metadata['name'] if 'name' in spec.metadata else i)
        
        sns.heatmap(np.array(values), vmin=0, vmax=1, cmap="viridis", annot=annot, ax=ax, xticklabels=axis_labels, yticklabels=axis_labels)
        plt.title(mode)


class Tmds(MassSpectrum):
    """
    A class for calculate TMDS spectrum

    Attributes
    ----------
    table: pandas Datarame
        tmds spectrum - mass, intensity and caclulatedd parameters
    """
    def __init__(self, table: pd.DataFrame = None) -> pd.DataFrame:
        super().__init__(table)

        if table is None:
            self.table = pd.DataFrame()
        else:
            self.table = table

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
            Minimum relative intensity for taking mass-difference
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
        tmds_spec['mass'] = unique
        tmds_spec['count'] = counts
        tmds_spec['intensity'] = tmds_spec['count']/mass_num
        tmds_spec = tmds_spec.sort_values(by='mass').reset_index(drop=True)

        value_zero = set([i/1000 for i in range (0, 300000)]) - set (unique)
        unique = np.append(unique, np.array(list(value_zero)))
        counts = np.append(counts, np.zeros(len(value_zero), dtype=float))

        peaks, properties = find_peaks(tmds_spec['intensity'], distance=wide, prominence=p/2)
        prob = []
        for peak in peaks:
            prob.append(tmds_spec.loc[peak-5:peak+5,'intensity'].sum())
        tmds_spec = tmds_spec.loc[peaks].reset_index(drop=True)
        tmds_spec['intensity'] = prob
        tmds_spec = tmds_spec.loc[tmds_spec['intensity'] > p]

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
        diff_spec['mass'] = unique
        diff_spec['count'] = counts
        diff_spec['intensity'] = diff_spec['count']/massl
        diff_spec = diff_spec.sort_values(by='mass').reset_index(drop=True)

        return Tmds(diff_spec)

    def assign(
        self,
        generated_bruttos_table: pd.DataFrame = None,
        error: float = 0.001,
        brutto_dict:dict = None,
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
        brutto_dict: dict
            Optional, default {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}
            generate brutto table if generated_bruttos_table is None.
        max_num: int
            Optional. Default 100
        
        Return
        ------
        Tmds object with assigned signals and elements
        """

        if brutto_dict is None:
            brutto_dict = {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}

        if generated_bruttos_table is None:
            generated_bruttos_table = brutto_gen(brutto_dict, rules=False)
            generated_bruttos_table = generated_bruttos_table.loc[generated_bruttos_table['mass'] > 0]

        res = super().assign(generated_bruttos_table=generated_bruttos_table, abs_error=error, sign='0').drop_unassigned().table

        if max_num is not None and len(res) > max_num:
            res = res.sort_values(by='intensity', ascending=False).reset_index(drop=True)
            res = res.loc[:max_num].reset_index(drop=True)
            res = res.sort_values(by='mass').reset_index(drop=True)
        
        return Tmds(res)


if __name__ == '__main__':
    pass