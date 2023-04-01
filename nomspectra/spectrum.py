#    Copyright 2019-2021 Rukhovich Gleb
#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nomspectra. 
#
#    nomspectra is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomspectra is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomspectra.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
import os
from typing import Callable, Dict, Sequence, Union, Optional, Mapping, Tuple
from functools import wraps
import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from scipy.optimize import curve_fit

from .brutto import brutto_gen, elements_table, get_elements_masses
from .metadata import MetaData


class Spectrum(object):
    """ 
    A class used to represent mass spectrum
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

        if table is not None:
            self.table = table
        else:
            self.table = pd.DataFrame(columns=["mass", 'intensity', "brutto", "calc_mass", "rel_error"])

        self.metadata = MetaData(metadata)

    @staticmethod
    def read_csv(
        filename: Union[Path, str],
        mapper: Optional[Mapping[str, str]] = None,
        ignore_columns: Optional[Sequence[str]] = None,
        take_columns: Optional[Sequence[str]] = None,
        take_only_mz: bool = False,
        sep: str = ",",
        intens_min: Optional[Union[int, float]] =  None,
        intens_max: Optional[Union[int, float]] = None,
        mass_min: Optional[Union[int, float]] =  None,
        mass_max: Optional[Union[int, float]] = None,
        assign_mark: bool = False,
        metadata: Optional[Dict] = None
    ) -> "Spectrum":
        """
        Read mass spectrum table from csv (Comma-Separated Values) file

        All parameters is optional except filename
        File must have header and at last two main columns: mass and intensity

        Parameters
        ----------
        filename: str
            path to mass spectrum table
        mapper: dict
            dictonary for recognize columns in mass spectrum file. 
            Example: {'m/z':'mass','I':'intensity'}
        ignore_columns: Sequence[str]
            list with names of columns that willn't loaded.
            if None load all columns.
            Example: ["index", "s/n"]
        take_columns: Sequence[str]
            list with names of columns that will be loaded, other will be ignored
            if None load all columns.
            Example: ["mass", "intensity", "C", "H", "N", "O"]
        take_only_mz: bool
            Load only mass and intesivity columns
        sep: str
            separator in mass spectrum table, \\t - for tab.
        intens_min: numeric
            bottom limit for intensity.
            by default it is None and don't restrict by this.
        intens_max: numeric
            upper limit for intensivity.
            by default it is None and don't restrict by this
        mass_min: numeric
            bottom limit for m/z.
            by default it is None and don't restrict by this
        mass_max: numeric
            upper limit for m/z.
            by default it is None and don't restrict by this
        assign_mark: bool
            default False. Mark peaks as assigned if they have elements.
            Need for load mass-list treated by external software
        metadata: Dict
            Optional. Default None. Metadata object that consist dictonary of metadata.
            if name not in metadata - name will take from filename.

        Return
        ------
        Spectrum
        """

        table = pd.read_csv(filename, sep=sep)
        if mapper:
            table = table.rename(columns=mapper)

        if take_columns:
            table = table.loc[:,take_columns]

        if ignore_columns:
            table = table.drop(columns=ignore_columns)

        if take_only_mz:
            table = table.loc[:,['mass','intensity']]

        if intens_min is not None:
            table = table.loc[table['intensity']>intens_min]

        if intens_max is not None:
            table = table.loc[table['intensity']<intens_max]

        if mass_min is not None:
            table = table.loc[table['mass']>mass_min]

        if mass_max is not None:
            table = table.loc[table['mass']<mass_max]

        table = table.sort_values(by="mass").reset_index(drop=True)

        if metadata is None:
            metadata = {}

        if 'name' not in metadata:
            head, tail = os.path.split(filename)
            metadata['name'] = tail.split('.')[0]

        res = Spectrum(table=table, metadata=metadata)

        if assign_mark:
            res._mark_assigned_by_brutto()

        return res
    
    def to_csv(self, 
                filename: Union[Path, str], 
                sep: str = ",") -> None:
        """
        Save Spectrum mass-list to csv file
        
        Parameters
        ----------
        filename: str
            Path for saving mass spectrum table with calculation
        sep: str
            Optional. Separator in saved file. By default it is ','

        Caution
        -------
        Metadata will be lost. For save them too use to_json method     
        """

        self.table.to_csv(filename, sep=sep, index=False)

    @staticmethod
    def read_json(filename: Union[Path, str]) -> "Spectrum":
        """
        Read mass spectrum from json own format

        Parameters
        ----------
        filename: str
            path to mass spectrum json file

        Return
        ------
        Spectrum
        """

        with open(filename, 'rb') as data:
            res = json.load(data)[0]
        table = pd.DataFrame(res['table'])
        metadata=res['metadata']

        return Spectrum(table=table, metadata=metadata)

    def to_json(self, filename: Union[Path, str]) -> None:
        """
        Save Spectrum mass-list to json format
        
        Parameters
        ----------
        filename: str
            Path for saving mass spectrum table with calculation to json file
        """

        out = {'metadata':copy.deepcopy(dict(self.metadata))}
        out['table'] = self.table.to_dict()
        with open(filename, 'w') as f:
            json.dump([out], f)

    def find_elements(self) -> Sequence[str]:
        """ 
        Find elements from columns of mass spectrum table.

        For example, column 'C' will be recognised as carbon 12C, column 'C_13" as 13C

        Returns
        -------
        list
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

        self.metadata.add({'elems':elems})

        return elems

    def _mark_assigned_by_brutto(self) -> None:
        """
        Mark peaks in loaded mass list if they have brutto

        Return
        ------
        Spectrum
        """

        assign = []
        elems = self.find_elements()
        for i, row in self.table.iterrows():
            flag = False
            for el in elems:
                if row[el] > 0:
                    flag = True
            assign.append(flag) 
        self.table['assign'] = assign

    def assign(
            self,
            brutto_dict: Optional[dict] = None,
            generated_bruttos_table: Optional[pd.DataFrame] = None,
            rel_error: Optional[float] = None,
            abs_error: Optional[float] = None,
            sign: str ='-',
            mass_min: Optional[float] =  None,
            mass_max: Optional[float] = None,
            intensity_min: Optional[float] =  None,
            intensity_max: Optional[float] = None,
            charge_max: int = 1
    ) -> "Spectrum":
        """
        Assigning brutto formulas to signal by mass
        
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
        intensity_min: float
            Optional. Default None. Minimall intensity for assigment
        intensity_max: float
            Optional. Default None. Maximum intensity for assigment
        charge_max: int
            Maximum charge in m/z. Default 1.   

        Return
        ------
        Spectrum 
        """

        if generated_bruttos_table is None:
            generated_bruttos_table = brutto_gen(brutto_dict)

        if mass_min is None:
            mass_min = self.table['mass'].min()
        if mass_max is None:
            mass_max = self.table['mass'].max()
        if intensity_min is None:
            intensity_min = self.table['intensity'].min()
        if intensity_max is None:
            intensity_max = self.table['intensity'].max()
        
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

        self.table = self.table.loc[:,['mass', 'intensity']].reset_index(drop=True)
        table = self.table.copy()

        masses = generated_bruttos_table["mass"].values

        elems = list(generated_bruttos_table.drop(columns=["mass"]))
        bruttos = generated_bruttos_table[elems].values.tolist()

        res = []
        for index, row in table.iterrows():

            if (row["mass"] < mass_min or 
                row["mass"] > mass_max or
                row["intensity"] < intensity_min or 
                row["intensity"] > intensity_max):
                res.append({"assign": False})
                continue 
            
            for charge in range(1, charge_max + 1):
                mass = (row["mass"] + mass_shift) * charge
                idx = np.searchsorted(masses, mass, side='left')
                if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                    idx -= 1

                if rel:
                    if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error/charge:
                        res.append({**dict(zip(elems, bruttos[idx])), "assign": True, "charge": charge})
                        break
                else:
                    if np.fabs(masses[idx] - mass) <= abs_error/charge:
                        res.append({**dict(zip(elems, bruttos[idx])), "assign": True, "charge": charge})
                        break
            else:
                res.append({"assign": False, "charge": 1})

        res = pd.DataFrame(res)

        table = table.join(res)
        self.table = self.table.merge(table, how='outer', on=list(self.table.columns))
        self.table['assign'] = self.table['assign'].fillna(False)
        self.table['charge'] = self.table['charge'].fillna(1)

        return self

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

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            self = copy.deepcopy(self)
            args = tuple([copy.deepcopy(arg) if isinstance(arg, Spectrum) else arg for arg in args])
            kwargs = {k: copy.deepcopy(v) if isinstance(v, Spectrum) else v for k, v in kwargs.items()}
            return func(self, *args, **kwargs)
        return wrapped

    @_copy
    def noise_filter(self,
                    force: float = 1.5,
                    intensity: Optional[float] = None,
                    quantile: Optional[float] = None 
                    ) -> 'Spectrum':
        """
        Remove noise from spectrum

        Parameters
        ----------
        intensity: float
            Cut by min intensity. 
            Default None and dont apply.
        quantile: float
            Cut by quantile. For example 0.1 mean that 10% 
            of peaks with minimal intensity will be cutted. 
            Default None and dont aplly
        force: float
            How many peaks should cut when auto-search noise level.
            Default 1.5 means that peaks with intensity more 
            than noise level*1.5 will be cutted
        
        Return
        ------
        Spectrum

        Caution
        -------
        There is risk of loosing data. Do it cautiously.
        Level of noise may be determenided wrong. 
        Draw and watch spectrum.
        """
        
        if intensity is not None:
            self.table = self.table.loc[self.table['intensity'] > intensity].reset_index(drop=True)
            self.metadata.add({'noise filter':f'intensity {intensity}'})
        
        elif quantile is not None:
            tresh = self.table['intensity'].quantile(quantile)
            self.table = self.table.loc[self.table['intensity'] > tresh].reset_index(drop=True)
            self.metadata.add({'noise filter':f'quantile {quantile}'})
        
        else:

            intens = self.table['intensity'].values
            cut_diapasone=np.linspace(0, np.mean(intens),100)

            d = []
            for i in cut_diapasone:
                d.append(len(intens[intens > i]))

            dx = np.gradient(d, 1)
            tresh = np.where(dx==np.min(dx))
            cut = cut_diapasone[tresh[0][0]] * force
            self.table = self.table.loc[self.table['intensity'] > cut].reset_index(drop=True)

            self.metadata.add({'noise filter':f'force {force}'})

        return self

    @_copy
    def drop_unassigned(self) -> "Spectrum":
        """
        Drop unassigned by brutto rows

        Return
        ------
        Spectrum

        Caution
        -------
        Danger of lose data - with these operation we exclude data that can be usefull
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        self.table = self.table.loc[self.table["assign"] == True].reset_index(drop=True)
        self.metadata.add({'drop_unassigned':True})

        return self

    @_copy
    def merge_duplicates(self) -> "Spectrum":
        """
        merge duplicataes with the same calculated mass with sum intensity

        Return
        ------
        Spectrum
        """
        if 'calc_mass' not in self.table.columns:
            self = self.calc_mass()

        cols = {col: ('sum' if col=='intensity' else 'max') for col in self.table.columns}
        self.table = self.table.groupby(['calc_mass'],as_index = False).agg(cols)
        return self

    @_copy
    def filter_by_C13(
        self, 
        rel_error: float = 0.5,
        remove: bool = False,
    ) -> 'Spectrum':
        """ 
        Check if peaks have the same brutto with C13 isotope

        Parameters
        ----------
        rel_error: float
            Optional. Default 0.5.
            Allowable ppm error when checking c13 isotope peak
        remove: bool
            Optional, default False. 
            Drop unassigned peaks and peaks without C13 isotope
        
        Return
        ------
        Spectrum
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
            self.metadata.add({'filter_C13':True})

        return self

    @_copy
    def normalize(self, how:str='sum') -> 'Spectrum':
        """
        Intensity normalize by intensity

        Parameters
        ----------
        how: {'sum', 'max', 'median', 'mean'}
            'sum' for normilize by sum of intensity of all peaks. (default)
            'max' for normilize by higher intensity peak.
            'median' for normilize by median of peaks intensity.
            'mean' for normilize by mean of peaks intensity.

        Return
        ------
        Spectrum
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
        
        self.metadata.add({'normilize':how})

        return self

    @_copy
    def merge_isotopes(self) -> "Spectrum":
        """
        Merge isotopes.

        For example if specrum list have 'C' and 'C_13' they will be summed in 'C' column.

        Return
        ------
        Spectrum

        Caution
        -------
        Danger of lose data - with these operation we exclude data that can be usefull       
        """

        elems = self.find_elements()
        for el in elems:
            res = el.split('_')
            if len(res) == 2:
                if res[0] not in self.table:
                    self.table[res[0]] = 0
                self.table[res[0]] = self.table[res[0]] + self.table[el]
                self.table = self.table.drop(columns=[el])
        
        self.metadata.add({'merge_isotopes':True})

        return self

    def copy(self) -> 'Spectrum':
        """
        Deepcopy of self Spectrum object

        Return
        ------
        Spectrum
        """

        table = copy.deepcopy(self.table)
        metadata = copy.deepcopy(self.metadata)

        return Spectrum(table = table, metadata = metadata)

    @_copy
    def calc_mass(self) -> "Spectrum":
        """
        Calculate mass from assigned brutto formulas and elements exact masses

        Add column "calc_mass" to self.table

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")
        
        elems = self.find_elements()
        
        table = self.table.loc[:,elems].copy()
        
        masses = get_elements_masses(elems)

        self.table["calc_mass"] = table.multiply(masses).sum(axis=1)
        self.table["calc_mass"] = np.round(self.table["calc_mass"], 6)
        self.table.loc[self.table["calc_mass"] == 0, "calc_mass"] = np.NaN

        return self

    @_copy
    def _calc_sign(self) -> str:
        """
        Determine sign from mass and calculated mass

        '-' for negative mode
        '+' for positive mode
        '0' for neutral

        Return
        ------
        str            
        """

        self = self.drop_unassigned()

        value = (self.table["calc_mass"]/self.table["charge"] - self.table["mass"]).mean()
        value = np.round(value,4)
        if value > 1:
            return '-'
        elif value > 0.0004 and value < 0.01:
            return '+'
        else:
            return '0'

    @_copy
    def calc_error(self, sign: Optional[str] = None) -> "Spectrum":
        """
        Calculate relative and absolute error of assigned peaks from measured and calculated masses

        Add columns "abs_error" and "rel_error" to self.table

        Parameters
        ----------
        sign: {'-', '+', '0'}
            Optional. Default None and get from metatdata or calculated by self. 
            Mode in which mass spectrum was gotten. 
            '-' for negative mode
            '+' for positive mode
            '0' for neutral
        
        Return
        ------
        Spectrum
        """

        if "calc_mass" not in self.table:
            self = self.calc_mass()

        if "charge" not in self.table.columns:
            self.table["charge"] = 1

        if sign is None:
            if 'sign' in self.metadata:
                sign = self.metadata['sign']
            else:
                sign = self._calc_sign()

        if sign == '-':
            self.table["abs_error"] = ((self.table["mass"] + (- 0.00054858 + 1.007825)) * self.table["charge"]) - self.table["calc_mass"] #-electron + proton
        elif sign == '+':
            self.table["abs_error"] = ((self.table["mass"] + 0.00054858) * self.table["charge"]) - self.table["calc_mass"]#+electron
        elif sign == '0':
            self.table["abs_error"] = (self.table["mass"] * self.table["charge"]) - self.table["calc_mass"]
        else:
            raise ValueError('Sended sign or sign in metadata is not correct. May be "+","-","0"')
        
        self.table["rel_error"] = self.table["abs_error"] / self.table["mass"] * 1e6
        
        return self

    ######################################################
    #Operation with two Spectrum
    ######################################################

    @_copy
    def __or__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function 'or' for two Spectrum object

        Return
        ------
        Spectrum
        """
        
        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        a = self.table.dropna()
        b = other.table.dropna()
        
        merged_df = pd.merge(a, b, on="calc_mass", how="outer", sort=True)
        merged_df["intensity"] = merged_df[["intensity_x", "intensity_y"]].mean(axis=1)
        merged_df = merged_df.drop(["intensity_x", "intensity_y"], axis=1)
        for col in merged_df.columns:
            if col[-2:] == '_x':
                merged_df[col[:-2]] = np.where(merged_df[f'{col[:-2]}_x'].isnull(), merged_df[f'{col[:-2]}_y'], 
                                np.where(merged_df[f'{col[:-2]}_y'].isnull(), merged_df[f'{col[:-2]}_x'], merged_df[f'{col[:-2]}_x']))
                merged_df = merged_df.drop([col, f'{col[:-2]}_y'], axis=1)

        metadata = {'operate':'or', 'name':MetaData.combine_two_name(self,other)}

        return Spectrum(table = merged_df, metadata=metadata)

    @_copy
    def __xor__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function 'xor' for two Spectrum object

        Return
        ------
        Spectrum
        """

        sub1 = self.__sub__(other)
        sub2 = other.__sub__(self)

        a = sub1.__or__(sub2)
        
        metadata = {'operate':'xor', 'name':MetaData.combine_two_name(self,other)}
        a.metadata = MetaData(metadata)
        
        return a

    @_copy
    def __and__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function 'and' for two Spectrum object

        Return
        ------
        Spectrum
        """

        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        a = self.table.dropna()
        b = other.table.dropna()
        
        merged_df = pd.merge(a, b, on="calc_mass", how="inner", sort=True)
        merged_df["intensity"] = merged_df[["intensity_x", "intensity_y"]].mean(axis=1)
        merged_df = merged_df.drop(["intensity_x", "intensity_y"], axis=1)
        for col in merged_df.columns:
            if col[-2:] == '_x':
                merged_df[col[:-2]] = np.where(merged_df[f'{col[:-2]}_x'].isnull(), merged_df[f'{col[:-2]}_y'], 
                                np.where(merged_df[f'{col[:-2]}_y'].isnull(), merged_df[f'{col[:-2]}_x'], merged_df[f'{col[:-2]}_x']))
                merged_df = merged_df.drop([col, f'{col[:-2]}_y'], axis=1)
        
        metadata = {'operate':'and', 'name':MetaData.combine_two_name(self,other)}

        return Spectrum(table = merged_df, metadata=metadata)
    
    def __add__(self, other: "Spectrum") -> "Spectrum":
        """
        addition self spectrum with other spectrum

        Return
        ------
        Spectrum
        """

        return self.__or__(other)

    @_copy
    def __sub__(self, other:"Spectrum") -> "Spectrum":
        """
        Substraction other spectrum from self spectrum

        Return
        ------
        Spectrum
        """
        
        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        a = self.table['calc_mass'].dropna().to_list()
        b = other.table['calc_mass'].dropna().to_list()
        
        operate = set(a) - set(b)

        mark = []
        res = copy.deepcopy(self.table)
        for i, row in res.iterrows():
            if row['calc_mass'] in operate:
                mark.append(row['calc_mass'])
            else:
                mark.append(np.NaN)
        res['calc_mass'] = mark
        res = res.dropna()

        metadata = {'operate':'sub', 'name':MetaData.combine_two_name(self,other)}

        return Spectrum(table = res, metadata=metadata)

    @_copy
    def intens_sub(self, other:"Spectrum") -> "Spectrum":
        """
        Substruction of other spectrum from self by intensivity

        Result Contain only peaks that higher than in other. 
        And intensity of this peaks is substraction of self and other.

        Parameters
        ----------
        other: Spectrum object
            other mass-scpectrum

        Return
        ------
        Spectrum 
        """

        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        #find common masses
        m = self & other
        msc = m.table['calc_mass'].values

        #extract table with common masses
        massE = self.table['calc_mass'].values
        rE = self.table[np.isin(massE, msc)]
        massL = other.table['calc_mass'].values
        rL = other.table[np.isin(massL, msc)]

        #substract intensity each others
        rE = rE.copy()
        rE['intensity'] = rE['intensity'] - rL['intensity']
        rE = rE.loc[rE['intensity'] > 0]
        
        #and add only own molecules
        res = (self - other) + Spectrum(rE)

        metadata = {'operate':'intens_sub', 'name':MetaData.combine_two_name(self,other)}
        res.metadata = MetaData(metadata)

        return res

    @_copy
    def simmilarity(self, other: "Spectrum", mode: Union[str, Callable] = 'cosine', func = None) -> float:
        """
        Calculate Simmilarity of self spectrum with other spectrum

        Parameters
        ----------
        other: Spectrum object
            second MaasSpectrum object with that calc simmilarity
        mode: {"tanimoto", "jaccard", "cosine"} or Function
            one of the simple simmilarity functions
            Mode can be: "tanimoto", "jaccard", "cosine". Default cosine.
            May also send here function, that will be 
            applayed to two pandas DataFrame with Spectrum data             

        Return
        ------
        float
        """

        self = self.normalize()
        other = other.normalize()

        if 'calc_mass' not in self.table:
            self = self.calc_mass()
        if 'calc_mass' not in other.table:
            other = other.calc_mass()

        s1 = self.drop_unassigned().normalize(how='sum')
        s2 = other.drop_unassigned().normalize(how='sum')

        df1 = pd.DataFrame()
        df1['cmass'] = s1.drop_unassigned().table['calc_mass']
        df1['intens'] = s1.drop_unassigned().table['intensity']

        df2 = pd.DataFrame()
        df2['cmass'] = s2.drop_unassigned().table['calc_mass']
        df2['intens'] = s2.drop_unassigned().table['intensity']

        res = df1.merge(df2, how='outer', on='cmass')
        res.fillna(0, inplace=True)

        a = res['intens_x'].values
        b = res['intens_y'].values

        a = a/np.sum(a)
        b = b/np.sum(b)      

        if isinstance(mode, str):
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
        else:
            return func(self.table, other.table)

    ###########################################
    # Calculation methods for brutto formulas #
    ###########################################

    @_copy
    def brutto(self) -> 'Spectrum':
        """
        Calculate string with brutto from assign table

        Add column "britto" to self.table

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        elems = self.find_elements()
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

    @_copy
    def cram(self) -> "Spectrum":
        """
        Mark rows that fit CRAM conditions
        (carboxylic-rich alicyclic molecules)

        Add column "CRAM" to self.table

        Return
        ------
        Spectrum

        References
        ----------
        Hertkorn, N. et al. Characterization of a major 
        refractory component of marine dissolved organic matter.
        Geochimica et. Cosmochimica Acta 70, 2990-3010 (2006)
        """

        if "DBE" not in self.table:
            self = self.dbe()        

        def check(row):
            if row['DBE']/row['C'] < 0.3 or row['DBE']/row['C'] > 0.68:
                return False
            if row['DBE']/row['H'] < 0.2 or row['DBE']/row['H'] > 0.95:
                return False
            if row['O'] == 0:
                return False
            elif row['DBE']/row['O'] < 0.77 or row['DBE']/row['O'] > 1.75:
                return False
            return True

        table = self.copy().merge_isotopes().table
        self.table['CRAM'] = table.apply(check, axis=1)

        return self

    @_copy
    def ai(self) -> 'Spectrum':
        """
        Calculate AI (aromaticity index)

        Add column "AI" to self.table

        Return
        ------
        Spectrum

        References
        ----------
        Koch, Boris P., and T. Dittmar. "From mass to structure: An aromaticity 
        index for high resolution mass data of natural organic matter." 
        Rapid communications in mass spectrometry 20.5 (2006): 926-932.
        """

        if "DBE_AI" not in self.table:
            self = self.dbe_ai()

        if "CAI" not in self.table:
            self = self.cai()

        self.table["AI"] = self.table["DBE_AI"] / self.table["CAI"]

        clear  = self.table["AI"].values[np.isfinite(self.table["AI"].values)]
        self.table['AI'] = self.table['AI'].replace(-np.inf, np.min(clear))
        self.table['AI'] = self.table['AI'].replace(np.inf, np.max(clear))
        self.table['AI'] = self.table['AI'].replace(np.nan, np.mean(clear))

        return self

    @_copy
    def cai(self) -> 'Spectrum':
        """
        Calculate CAI (C - O - N - S - P)

        Add column "CAI" to self.table

        Return
        ------
        Spectrum
        """
        
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.merge_isotopes().table

        for element in "CONSP":
            if element not in table:
                table[element] = 0

        self.table['CAI'] = table["C"] - table["O"] - table["N"] - table["S"] - table["P"]

        return self

    @_copy
    def dbe_ai(self) -> 'Spectrum':
        """
        Calculate DBE_AI (1 + C - O - S - 0.5 * (H + N + P))

        Add column "DBE_AI" to self.table

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.merge_isotopes().table

        for element in "CHONPS":
            if element not in table:
                table[element] = 0

        self.table['DBE_AI'] = 1.0 + table["C"] - table["O"] - table["S"] - 0.5 * (table["H"] + table['N'] + table["P"])

        return self

    @_copy
    def dbe(self) -> 'Spectrum':
        """
        Calculate DBE (1 + C - 0.5 * (H + N))

        Add column "DBE" to self.table

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.merge_isotopes().table

        for element in "CHON":
            if element not in table:
                table[element] = 0

        self.table['DBE'] = 1.0 + table["C"] - 0.5 * (table["H"] - table['N'])

        return self

    @_copy
    def dbe_o(self) -> 'Spectrum':
        """
        Calculate DBE - O

        Add column "DBE-O" to self.table

        Return
        ------
        Spectrum 
        """

        if "DBE" not in self.table:
            self = self.dbe()

        table = self.merge_isotopes().table
        self.table['DBE-O'] = table['DBE'] - table['O']

        return self

    @_copy
    def dbe_oc(self) -> 'Spectrum':
        """
        Calculate (DBE - O) / C

        Add column "DBE-OC" to self.table

        Return
        ------
        Spectrum
        """

        if "DBE" not in self.table:
            self = self.dbe()

        table = self.merge_isotopes().table
        self.table['DBE-OC'] = (table['DBE'] - table['O'])/table['C']

        return self

    @_copy
    def hc_oc(self) -> 'Spectrum':
        """
        Calculate H/C and O/C

        Add columns "H/C" and "O/C" to self.table

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.merge_isotopes().table
        self.table['H/C'] = table['H']/table['C']
        self.table['O/C'] = table['O']/table['C']

        return self

    @_copy
    def kendrick(self) -> 'Spectrum':
        """
        Calculate Kendrick mass and Kendrick mass defect

        Add columns "Ke" and 'KMD" to self.table

        Return
        ------
        Spectrum
        """

        if 'calc_mass' not in self.table:
            self = self.calc_mass()

        self.table['Ke'] = self.table['calc_mass'] * 14/14.01565
        self.table['KMD'] = np.floor(self.table['calc_mass'].values) - np.array(self.table['Ke'].values)
        self.table.loc[self.table['KMD']<=0, 'KMD'] = self.table.loc[self.table['KMD']<=0, 'KMD'] + 1

        return self

    @_copy
    def nosc(self) -> 'Spectrum':
        """
        Calculate Normal oxidation state of carbon (NOSC)

        Add column "NOSC" to self.table

        Notes
        -----
        >0 - oxidate state.
        <0 - reduce state.
        0 - neutral state

        References
        ----------
        Boye, Kristin, et al. "Thermodynamically 
        controlled preservation of organic carbon 
        in floodplains."
        Nature Geoscience 10.6 (2017): 415-419.

        Return
        ------
        Spectrum
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.merge_isotopes().table

        for element in "CHONS":
            if element not in table:
                table[element] = 0

        self.table['NOSC'] = 4.0 - (table["C"] * 4 + table["H"] - table['O'] * 2 - table['N'] * 3 - table['S'] * 2)/table['C']

        return self
    
    @_copy
    def mol_class(self, how: Optional[str] = None) -> "Spectrum":
        """
        Assign molecular class for formulas

        Add column "class" to self.table

        Parameters
        ----------
        how: {'kellerman', 'perminova', 'laszakovits'}
            How devide to calsses. Optional. Default 'laszakovits'

        Return
        ------
        Spectrum

        References
        ----------
        Laszakovits, J. R., & MacKay, A. A. Journal of the American Society for Mass Spectrometry, 2021, 33(1), 198-202.
        A. M. Kellerman, T. Dittmar, D. N. Kothawala, L. J. Tranvik. Nat. Commun. 2014, 5, 3804
        Perminova I. V. Pure and Applied Chemistry. 2019. Vol. 91, № 5. P. 851-864
        """

        if 'AI' not in self.table:
            self = self.ai()
        if 'H/C' not in self.table or 'O/C' not in self.table:
            self = self.hc_oc()

        table = self.merge_isotopes().table

        for element in "CHON":
            if element not in table:
                table[element] = 0

        def get_zone_kell(row):

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
        
        def get_zone_perm(row):

            if row['O/C'] < 0.5:
                if row['H/C'] < 1:
                    return 'condensed_tanins'
                elif row['H/C'] < 1.4:
                    return 'phenylisopropanoids'
                elif row['H/C'] < 1.8:
                    return 'terpenoids'
                elif row['H/C'] <= 2.2:
                    if row['O/C'] < 0.25:
                        return 'lipids'
                    else:
                        return 'proteins'
                else:
                    return 'undefinded'
            elif row['O/C'] <= 1:
                if row['H/C'] < 1.4:
                    return 'hydrolyzable_tanins'
                elif row['H/C'] <= 2.2:
                    return 'carbohydrates'
                else:
                    return 'undefinded'
            else:
                return 'undefinded'

        def get_zone_lasz(row):
            if row['H/C'] >= 0.86 and row['H/C'] <=1.34 and row['O/C'] >= 0.21 and row['O/C'] <=0.44:
                return 'lignin'
            elif row['H/C'] >= 0.7 and row['H/C'] <=1.01 and row['O/C'] >= 0.16 and row['O/C'] <=0.84:
                return 'tannin'
            elif row['H/C'] >= 1.33 and row['H/C'] <=1.84 and row['O/C'] >= 0.17 and row['O/C'] <=0.48:
                return 'peptide'
            elif row['H/C'] >= 1.34 and row['H/C'] <=2.18 and row['O/C'] >= 0.01 and row['O/C'] <=0.35:
                return 'lipid'
            elif row['H/C'] >= 1.53 and row['H/C'] <=2.2 and row['O/C'] >= 0.56 and row['O/C'] <=1.23:
                return 'carbohydrate'
            elif row['H/C'] >= 1.62 and row['H/C'] <=2.35 and row['O/C'] >= 0.56 and row['O/C'] <=0.95:
                return 'aminosugar'
            else:
                return 'undefinded'
        
        if how == 'perminova':
            self.table['class'] = table.apply(get_zone_perm, axis=1)
        elif how == 'kellerman':
            self.table['class'] = table.apply(get_zone_kell, axis=1)
        else:
            self.table['class'] = table.apply(get_zone_lasz, axis=1)

        return self

    @_copy
    def get_mol_class(self, how_average: str = "weight", how: Optional[str] = None) -> pd.DataFrame:
        """
        get molercular class density

        Parameters
        ----------
        how_average: {'weight', 'count'}
            how average density. Default "weight" - weight by intensity.
            Also can be "count".
        how: {'kellerman', 'perminova', 'laszakovits'}
            How devide to calsses. Optional. Default 'laszakovits'

        Return
        ------
        pandas Dataframe
        
        References
        ----------
        Laszakovits, J. R., & MacKay, A. A. Journal of the American Society for Mass Spectrometry, 2021, 33(1), 198-202.
        A. M. Kellerman, T. Dittmar, D. N. Kothawala, L. J. Tranvik. Nat. Commun. 5, 3804 (2014)
        Perminova I. V. Pure and Applied Chemistry. 2019. Vol. 91, № 5. P. 851-864
        """

        self = self.drop_unassigned().mol_class(how=how)
        count_density = len(self.table)
        sum_density = self.table["intensity"].sum()

        out = []

        if how == 'perminova':
            zones = ['condensed_tanins',
                    'hydrolyzable_tanins',
                    'phenylisopropanoids',
                    'terpenoids',
                    'lipids',
                    'proteins',
                    'carbohydrates',
                    'undefinded']
        elif how == 'kellerman':
            zones = ['unsat_lowOC',
                    'unsat_highOC',
                    'condensed_lowOC',
                    'condensed_highOC',
                    'aromatic_lowOC',
                    'aromatic_highOC',
                    'aliphatics',            
                    'lipids',
                    'N-satureted',
                    'undefinded']
        else:
            zones = ['aminosugar',
                    'carbohydrate',
                    'lignin',
                    'lipid',
                    'peptide',
                    'tannin',
                    'undefinded']


        for zone in zones:

            if how_average == "count":
                out.append([zone, len(self.table.loc[self.table['class'] == zone])/count_density])

            elif how_average == "weight":
                out.append([zone, self.table.loc[self.table['class'] == zone, 'intensity'].sum()/sum_density])

            else:
                raise ValueError(f"how_average should be count or intensity not {how_average}")
        
        return pd.DataFrame(data=out, columns=['class', 'density'])

    @_copy
    def get_dbe_vs_o(self, 
                        olim: Optional[Tuple[int, int]] = None, 
                        draw: bool = True, 
                        ax: Optional[plt.axes] = None, 
                        **kwargs) -> Tuple[float, float]:
        """
        Calculate DBE vs nO by linear fit
        
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
        (float, float)
            a and b in fit DBE = a * nO + b

        References
        ----------
        Bae, E., Yeo, I. J., Jeong, B., Shin, Y., Shin, K. H., & Kim, S. (2011). 
        Study of double bond equivalents and the numbers of carbon and oxygen 
        atom distribution of dissolved organic matter with negative-mode FT-ICR MS.
        Analytical chemistry, 83(11), 4193-4199.
        """

        if 'DBE' not in self.table:
            self = self.dbe()
        
        self = self.drop_unassigned()
        if olim is None:
            no = list(range(int(self.table['O'].min())+5, int(self.table['O'].max())-5))
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

    @_copy
    def get_squares_vk(self,
                        how_average: str = 'weight',
                        ax: Optional[plt.axes] = None, 
                        draw: bool = False) -> pd.DataFrame:
        """
        Calculate density in Van Krevelen diagram divided into 20 squares

        Squares index in Van-Krevelen diagram if H/C is rows, O/C is columns:
        [[5, 10, 15, 20],
         [4, 9, 14, 19],
         [3, 8, 13, 18],
         [2, 7, 12, 17],
         [1, 6, 11, 16]]

        H/C divided by [0-0.6, 0.6-1, 1-1.4, 1.4-1.8, 1.8-2.2]
        O/C divided by [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]

        Parameters
        ----------
        how_average: {'weight', 'count'}
            How calculate average. My be "count" or "weight" (default)
        ax: matplotlib ax
            Optional. external ax
        draw: bool
            Optional. Default False. Plot heatmap

        Return
        ------
        Pandas Dataframe

        References
        ----------
        Perminova I. V. From green chemistry and nature-like technologies towards 
        ecoadaptive chemistry and technology // Pure and Applied Chemistry. 
        2019. Vol. 91, № 5. P. 851-864.
        """

        if 'H/C' not in self.table or 'O/C' not in self.table:
            self = self.hc_oc().drop_unassigned()

        d_table = []
        sq = []

        for y in [ (1.8, 2.2), (1.4, 1.8), (1, 1.4), (0.6, 1), (0, 0.6)]:
            hc = []
            for x in  [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]:
                temp = copy.deepcopy(self)
                temp.table = temp.table.loc[(temp.table['O/C'] >= x[0]) & (temp.table['O/C'] < x[1]) & (temp.table['H/C'] >= y[0]) & (temp.table['H/C'] < y[1])]

                if how_average == 'count':
                    res = len(temp)/len(self)
                    hc.append(res)
                    sq.append(res)
                elif how_average == 'weight':
                    res = temp.table['intensity'].sum()/self.table['intensity'].sum()
                    hc.append(res)
                    sq.append(res)
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
        square = pd.DataFrame()
        square['value'] = sq
        square['square'] = [5,10,15,20,   4,9,14,19,   3,8,13,18,    2,7,12,17,   1,6,11,16]
        
        return square.sort_values(by='square').reset_index(drop=True)

    @_copy
    def get_mol_metrics(self, 
                        metrics: Optional[Sequence[str]] = None, 
                        func: Optional[str] = None) -> pd.DataFrame:
        """
        Get average metrics

        Parameters
        ----------
        metrics: Sequence[str]
            Optional. Default None. Chose metrics fot watch.
        func: {'weight', 'mean', 'median', 'max', 'min', 'std'}
            How calculate average. My be "weight" (default - weight average on intensity),
            "mean", "median", "max", "min", "std" (standard deviation)

        Return
        ------
        pandas DataFrame
        """

        self = self.calc_all_metrics().drop_unassigned().normalize()

        if metrics is None:
            metrics = set(self.table.columns) - set(['intensity', 'calc_mass', 'rel_error','abs_error',
                                                    'assign', 'charge', 'class', 'brutto', 'Ke', 'KMD'])

        res = []
        metrics = np.sort(np.array(list(metrics)))

        if func is None:
            func = 'weight'

        func_dict = {'mean': lambda col : np.average(self.table[col]),
                    'weight': lambda col : np.average(self.table[col], weights=self.table['intensity']),
                    'median': lambda col : np.median(self.table[col]),
                    'max': lambda col : np.max(self.table[col]),
                    'min': lambda col : np.min(self.table[col]),
                    'std': lambda col : np.std(self.table[col])}
        if func not in func_dict:
            raise ValueError(f'not correct value - {func}')
        else:
            f = func_dict[func]

        for col in metrics:
            try:
                res.append([col, f(col)])
            except:
                res.append([col, np.NaN])

        return pd.DataFrame(data=res, columns=['metric', 'value'])

    @_copy
    def calc_all_metrics(self) -> "Spectrum":
        """
        Calculated all available metrics

        Return
        ------
        Spectrum
        """

        self = self.calc_mass()
        self = self.calc_error()
        self = self.dbe()
        self = self.dbe_o()
        self = self.ai()
        self = self.dbe_oc()
        self = self.dbe_ai()
        self = self.mol_class()
        self = self.hc_oc()
        self = self.cai()
        self = self.cram()
        self = self.nosc()
        self = self.brutto()
        self = self.kendrick()

        return self

    #############################################################
    # passing pandas DataFrame methods for represent mass table #
    #############################################################

    def head(self, num: Optional[int] = None) -> pd.DataFrame:
        """
        Show head of mass spec table

        Parameters
        ----------
        num: int
            Optional. number of head string

        Return
        ------
        Pandas Dataframe
        """

        if num is None:
            return self.table.head()
        else:
            return self.table.head(num)

    def tail(self, num: Optional[int] = None) -> pd.DataFrame:
        """
        Show tail of Spectrum table

        Parameters
        ----------
        num: int
            Optional. number of tail string

        Return
        ------
        Pandas Dataframe
        """

        if num is None:
            return self.table.tail()
        else:
            return self.table.tail(num)

    def __len__(self) -> int:
        """
        Length of Spectrum table

        Return
        ------
        int - length of Spectrum table
        """

        return len(self.table)
    
    def __getitem__(self, item: Union[str, Sequence[str]]) -> pd.DataFrame:
        """
        Get items or slice from Spectrum table

        Return
        ------
        Pandas Dataframe
        """

        return self.table[item]

    def __repr__(self) -> str:
        """
        Representation of Spectrum object.

        Return
        ------
        str
        """

        columns = ["mass", 'intensity']
        return self.table[columns].__repr__()

    def __str__(self) -> str:
        """
        Representation of Spectrum object.

        Return
        ------
        str
        """

        columns = ["mass", 'intensity']
        return self.table[columns].__str__()


if __name__ == '__main__':
    pass