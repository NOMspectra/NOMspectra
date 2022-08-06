#    Copyright 2019-2021 Rukhovich Gleb
#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nhsmass. 
#
#    nhsmass is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nhsmass is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nhsmass.  If not, see <http://www.gnu.org/licenses/>.

from optparse import Option
from pathlib import Path
from typing import List, Dict, Sequence, Union, Optional, Mapping, Tuple
import copy
import json
from collections import UserList

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

from .brutto import brutto_gen, elements_table, get_elements_masses, gen_from_brutto
from .metadata import MetaData


class Spectrum(object):
    """ 
    A class used to represent mass spectrum

    Attributes
    ----------
    table : pandas Datarame
        Consist spectrum (mass and intensity of peaks) and all calculated parameters
        like brutto formulas, calculated mass, relative errorr
    metadata: MetaData
        Metadata object that consist dictonary of metadata
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

        self.features = ["mass", 'intensity', "calc_mass", 'intensity', "rel_error"]

        if table is not None:
            self.table = table
        else:
            self.table = pd.DataFrame(columns=["mass", 'intensity', "brutto", "calc_mass", "rel_error"])

        self.metadata = MetaData(metadata)

    def read_csv(
        self,
        filename: Union[Path, str],
        mapper: Optional[Mapping[str, str]] = None,
        ignore_columns: Optional[Sequence[str]] = None,
        take_columns: Optional[Sequence[str]] = None,
        take_only_mz: bool = False,
        sep: str = ",",
        intens_min: Optional[float] =  None,
        intens_max: Optional[float] = None,
        mass_min: Optional[float] =  None,
        mass_max: Optional[float] = None,
        assign_mark: bool = False,
        metadata: Optional[Dict] = None
    ) -> "Spectrum":
        """
        Read mass spectrum table from csv (Comma-Separated Values) file

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
        Spectrum object
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

        if metadata is not None:            
            self.metadata.add(metadata=metadata)

        #FIXME bad solution
        if 'name' not in self.metadata:
            self.metadata['name'] = filename.split('/')[-1].split('.')[0]

        return self
    
    def to_csv(self, 
                filename: Union[Path, str], 
                sep: str = ",") -> None:
        """
        Saves Spectrum mass-list to csv Spectrum
        
        Parameters
        ----------
        filename: str
            Path for saving mass spectrum table with calculation to csv file
        sep: str
            Optional. Separator in saved file. By default it is ','

        Caution
        -------
        Metadata will be lost. For save them too use save_json method     
        """
        self.table.to_csv(filename, sep=sep, index=False)

    def read_json(self, filename: Union[Path, str]) -> "Spectrum":
        """
        Read mass spectrum from json own format

        Parameters
        ----------
        filename: str
            path to mass spectrum json file, absoulute or relative

        Return
        ------
        Spectrum object
        """
        with open(filename, 'rb') as data:
            res = json.load(data)[0]
        self.table = pd.DataFrame(res['table'])
        self.metadata=res['metadata']

        return self

    def to_json(self, filename: Union[Path, str]) -> None:
        """
        Saves Spectrum mass-list to JSON own format
        
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

    def _mark_assigned_by_brutto(self) -> None:
        """Mark paeks in loaded mass list if they have brutto

        Return
        ------
        Spectrum object with assigned mark
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
    ) -> "Spectrum":
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
        intensity_min: float
            Optional. Default None. Minimall intensity for assigment
        intensity_max: float
            Optional. Default None. Maximum intensity for assigment   

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
        Remove noise from spectrum.
        Before applying spectrum will be normalized bu sum.

        Caution
        -------
        There is risk of loosing data. Do it cautiously.
        Level of noise may be determenided wrong. 
        Draw and watch spectrum.

        Parameters
        ----------
        intensity: float
            Simply cut directly by min intensity. 
            Default None and dont apply.
        quantile: float
            Cut by quantile. For example 0.1 mean that 10% 
            of peaks with minimal intensity will be cuted. 
            Default None and dont aplly
        force: float
            How many peaks should cut when auto-search noise level.
            Default 1.5 means that peaks with intensity more 
            than noise level*1.5 will be cutted
        
        Return
        ------
        Spectrum
        """
        self = self.normalize()
        
        if intensity is not None:
            self.table = self.table.loc[self.table['intensity'] > intensity].reset_index(drop=True)
        
        elif quantile is not None:
            tresh = self.table['intensity'].quantile(quantile)
            self.table = self.table.loc[self.table['intensity'] > tresh].reset_index(drop=True)
        
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

        return self

    @_copy
    def drop_unassigned(self) -> "Spectrum":
        """
        Drop unassigned mass from Mass Spectrum table

        Return
        ------
        Spectrum object that contain only assigned by brutto formulas peaks

        Caution
        -------
        Danger of lose data - with these operation we exclude data that can be usefull
        """

        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        self.table = self.table.loc[self.table["assign"] == True].reset_index(drop=True)

        return self

    @_copy
    def filter_by_C13(
        self, 
        rel_error: float = 0.5,
        remove: bool = False,
    ) -> 'Spectrum':
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
    def normalize(self, how:str='sum') -> 'Spectrum':
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
        Intensity normalized Spectrum instance
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

    @_copy
    def recallibrate(self, 
                    error_table: Optional["ErrorTable"] = None, 
                    how: str = 'assign') -> "Spectrum":
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
        Spectrum object with recallibrated mass
        '''
        if error_table is None:
            if how == 'assign':
                error_table = ErrorTable().assign_error(self)
            elif how == 'mdm':
                error_table = ErrorTable().massdiff_error(self)
            else:
                etalon = Spectrum().load(filename=how)
                error_table = ErrorTable().etalon_error(spec=self, etalon=etalon)

        err = copy.deepcopy(error_table.table)
        self.table = self.table.reset_index(drop=True)
        wide = len(err)

        min_mass = err['mass'].min()
        max_mass = err['mass'].max()
        a = np.linspace(min_mass, max_mass, wide+1)

        for i in range(wide):
            for ind in self.table.loc[(self.table['mass']>a[i]) & (self.table['mass']<=a[i+1])].index:
                mass = self.table.loc[ind, 'mass']
                e = mass * err.loc[i, 'ppm'] / 1000000
                self.table.loc[ind, 'mass'] = self.table.loc[ind, 'mass'] + e
                
        return self

    @_copy
    def assign_by_tmds (
        self, 
        tmds_spec: Optional["Tmds"] = None, 
        abs_error: float = 0.001,
        p = 0.2,
        max_num: Optional[int] = None,
        C13_filter: bool = True
        ) -> "Spectrum":
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
        Spectrum object new assign brutto formulas
        '''
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        if tmds_spec is None:
            tmds_spec = Tmds(spec=self).calc(p=p, C13_filter=C13_filter) #by varifiy p-value we can choose how much mass-diff we will take
            tmds_spec = tmds_spec.assign(max_num=max_num)
            tmds_spec = tmds_spec.calc_mass()

        tmds = tmds_spec.table.sort_values(by='intensity', ascending=False).reset_index(drop=True)
        tmds = tmds.loc[tmds['intensity'] > p].sort_values(by='mass', ascending=True).reset_index(drop=True)
        elem = tmds_spec.find_elements()

        assign_false = copy.deepcopy(self.table.loc[self.table['assign'] == False]).reset_index(drop=True)
        assign_true = copy.deepcopy(self.table.loc[self.table['assign'] == True]).reset_index(drop=True)
        masses = assign_true['mass'].values
        mass_dif_num = len(tmds)

        for i, row_tmds in tqdm(tmds.iterrows(), total=mass_dif_num):

            mass_shift = - row_tmds['calc_mass']
            
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

        #FIXME probably bad solutions
        assign_true = assign_true.append(assign_false, ignore_index=True).sort_values(by='mass').reset_index(drop=True)

        out = Spectrum(assign_true)
        out = out.calc_mass()

        out_false = out.table.loc[out.table['assign'] == False]
        out_true = out.table.loc[out.table['assign'] == True].drop_duplicates(subset="calc_mass")

        out2 = pd.merge(out_true, out_false, how='outer').reset_index(drop=True).sort_values(by='mass').reset_index(drop=True)
        self.table = out2
        
        return self

    @_copy
    def sum_isotopes(self) -> "Spectrum":
        """
        All isotopes will be sum and title as main.

        Return
        ------
        Spectrum object without minor isotopes        
        """

        elems = self.find_elements()
        for el in elems:
            res = el.split('_')
            if len(res) == 2:
                if res[0] not in self.table:
                    self.table[res[0]] = 0
                self.table[res[0]] = self.table[res[0]] + self.table[el]
                self.table = self.table.drop(columns=[el]) 

        return self

    def copy(self) -> 'Spectrum':
        """
        Deepcopy of self Spectrum object

        Return
        ------
        Deepcopy of self Spectrum object
        """
        return copy.deepcopy(Spectrum(self.table))

    @_copy
    def calc_mass(self) -> "Spectrum":
        """
        Calculate mass from assigned brutto formulas

        Return
        ------
        Spectrum object with calculated mass
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
        Calculate sign from mass and calculated mass

        Return
        ------
        str: "-", "+", "0"
        """
        self = self.drop_unassigned()

        value = (self.table["calc_mass"]- self.table["mass"]).mean()
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
        Spectrum object wit calculated error
        """
        if "calc_mass" not in self.table:
            self = self.calc_mass()

        if sign is None:
            if 'sign' in self.metadata:
                sign = self.metadata['sign']
            else:
                sign = self._calc_sign()

        if sign == '-':
            self.table["abs_error"] = self.table["mass"] - self.table["calc_mass"] + (- 0.00054858 + 1.007825) #-electron + proton
        elif sign == '+':
            self.table["abs_error"] = self.table["mass"] - self.table["calc_mass"] + 0.00054858 #+electron
        elif sign == '0':
            self.table["abs_error"] = self.table["mass"] - self.table["calc_mass"]
        else:
            raise Exception('Sended sign or sign in metadata is not correct. May be "+","-","0"')
        
        self.table["rel_error"] = self.table["abs_error"] / self.table["mass"] * 1e6
        
        return self

    ######################################################
    #Operation with two Spectrum
    ######################################################

    @_copy
    def __or__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function or for two Spectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        Spectrum object contain all assigned brutto formulas from two spectrum
        """
        
        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        a = self.table.dropna()
        b = other.table.dropna()
        
        a = a.append(b, ignore_index=True)
        a = a.drop_duplicates(subset=['calc_mass'])

        return Spectrum(a)

    @_copy
    def __xor__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function xor for two Spectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        Spectrum object contain xor assigned brutto formulas from two spectrum
        """

        sub1 = self.__sub__(other)
        sub2 = other.__sub__(self)
        
        return sub1.__or__(sub2)

    @_copy
    def __and__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function and for two Spectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        Spectrum object contain common assigned brutto formulas from two spectrum
        """

        self = self.normalize()
        other = other.normalize()

        if "calc_mass" not in self.table:
            self = self.calc_mass()
        if "calc_mass" not in other.table:
            other = other.calc_mass()

        a = self.table['calc_mass'].dropna().to_list()
        b = other.table['calc_mass'].dropna().to_list()
        
        operate = set(a) & set(b)

        mark = []
        res = copy.deepcopy(self.table)
        for i, row in res.iterrows():
            if row['calc_mass'] in operate:
                mark.append(row['calc_mass'])
            else:
                mark.append(np.NaN)
        res['calc_mass'] = mark
        res = res.dropna()

        return Spectrum(res)
    
    def __add__(self, other: "Spectrum") -> "Spectrum":
        """
        Logic function or for two Spectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        Spectrum object contain all assigned brutto formulas from two spectrum
        """
        return self.__or__(other)

    @_copy
    def __sub__(self, other:"Spectrum") -> "Spectrum":
        """
        Logic function substraction for two Spectrum object

        Work by calculated mass from brutto formulas

        Return
        ------
        Spectrum object contain substraction assigned brutto formulas from two spectrum
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

        return Spectrum(res)

    @_copy
    def intens_sub(self, other:"Spectrum") -> "Spectrum":
        """
        Calculate substruction by intensivity

        Parameters
        ----------
        other: Spectrum object
            other mass-scpectrum

        Return
        ------
        Spectrum object contain only that peak
        that higher than in other. And intensity of this peaks
        is substraction of self and other.
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
        return (self - other) + Spectrum(rE)  

    @_copy
    def simmilarity(self, other: "Spectrum", mode: str = 'cosine') -> float:
        """
        Calculate Simmilarity

        Parameters
        ----------
        other: Spectrum object
            second MaasSpectrum object with that calc simmilarity
        mode: str
            Optionaly. Default cosine. 
            one of the simmilarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"

        Return
        ------
        float Simmilarity index
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

    ###########################################
    # Calculation methods for brutto formulas #
    ###########################################

    @_copy
    def brutto(self) -> 'Spectrum':
        """
        Calculate brutto formulas from assign table

        Return
        ------
        Spectrum object wit calculated bruttos
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
            self = self.dbe()        

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
    def ai(self) -> 'Spectrum':
        """
        Calculate AI

        Return
        ------
        Spectrum object with calculated AI
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
        Calculate CAI

        Return
        ------
        Spectrum object with calculated CAI
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
    def dbe_ai(self) -> 'Spectrum':
        """
        Calculate DBE

        Return
        ------
        Spectrum object with calculated DBE
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
    def dbe(self) -> 'Spectrum':
        """
        Calculate DBE

        Return
        ------
        Spectrum object with calculated DBE
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
    def dbe_o(self) -> 'Spectrum':
        """
        Calculate DBE-O

        Return
        ------
        Spectrum object with calculated DBE-O
        """
        if "DBE" not in self.table:
            self = self.dbe()

        table = self.sum_isotopes().table
        self.table['DBE-O'] = table['DBE'] - table['O']

        return self

    @_copy
    def dbe_oc(self) -> 'Spectrum':
        """
        Calculate DBE-O/C

        Return
        ------
        Spectrum object with calculated DBE-O/C
        """
        if "DBE" not in self.table:
            self = self.dbe()

        table = self.sum_isotopes().table
        self.table['DBE-OC'] = (table['DBE'] - table['O'])/table['C']

        return self

    @_copy
    def hc_oc(self) -> 'Spectrum':
        """
        Calculate H/C and O/C

        Return
        ------
        Spectrum object with calculated H/C O/C
        """
        if "assign" not in self.table:
            raise Exception("Spectrum is not assigned")

        table = self.sum_isotopes().table
        self.table['H/C'] = table['H']/table['C']
        self.table['O/C'] = table['O']/table['C']

        return self

    @_copy
    def kendrick(self) -> 'Spectrum':
        """
        Calculate Kendrick mass and Kendrick mass defect

        Return
        ------
        Spectrum object with calculated Ke and KMD
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
        Calculate Normal oxidation state of carbon (NOSC).

        Notes
        -----
        >0 - oxidate state.
        <0 - reduce state.
        0 - neutral state

        Return
        ------
        Spectrum object with calculated DBE

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
    def mol_class(self) -> "Spectrum":
        """
        Assign molecular class for formulas

        Return
        ------
        Spectrum object with assigned zones
        """

        if 'AI' not in self.table:
            self = self.ai()
        if 'H/C' not in self.table or 'O/C' not in self.table:
            self = self.hc_oc()

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
    def get_mol_class(self, how_average: str = "weight") -> pd.DataFrame:
        """
        get molercular class density

        Parameters
        ----------
        how_average: str
            how average density. Default "weight" - weight by intensity.
            Also can be "count".

        Return
        ------
        pandas Dataframe with columns: mol_class, density
        
        References
        ----------
        Zherebker, Alexander, et al. "Interlaboratory comparison of 
        humic substances compositional space as measured by Fourier 
        transform ion cyclotron resonance mass spectrometry 
        (IUPAC Technical Report)." 
        Pure and Applied Chemistry 92.9 (2020): 1447-1467.
        """

        self = self.drop_unassigned().mol_class()
        count_density = len(self.table)
        sum_density = self.table["intensity"].sum()

        out = []
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
            self = self.dbe()

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

    def squares_vk(self, 
                   ax: Optional[plt.axes] = None, 
                   draw: bool = True) -> pd.DataFrame:
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
            self = self.hc_oc()   

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

    @_copy
    def calc_all_metrics(self) -> "Spectrum":
        """
        Calculated all avaible metrics

        Return
        ------
        Spectrum object with calculated metrics
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

    @_copy
    def get_mol_metrics(self, 
                        metrics: Optional[Sequence[str]] = None, 
                        how_average: str = 'weight') -> pd.DataFrame:
        """
        Get average metrics

        Parameters
        ----------
        metrics: Sequence[str]
            Optional. Default None. Chose metrics fot watch.
        how_average: str
            How calculate average. My be "count" or "weight" ((default))

        Return
        ------
        pd.DataFrame
        """

        self = self.calc_all_metrics().drop_unassigned().normalize()

        if metrics is None:
            metrics = set(self.table.columns) - set(['intensity', 'calc_mass', 'rel_error','abs_error',
                                                    'assign', 'class', 'brutto', 'Ke', 'KMD'])

        res = []
        metrics = np.sort(np.array(list(metrics)))

        for col in metrics:

            if how_average=='count':
                res.append([col, np.average(self.table[col])])
            
            elif how_average=='weight':
                res.append([col, np.average(self.table[col], weights=self.table['intensity'])])
            else:
                raise ValueError(f'how_average can be only "count" or "weight", not {how_average}')

        return pd.DataFrame(data=res, columns=['metric', 'value'])

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
        Pandas Dataframe head of MassSpec table
        """
        if num is None:
            return self.table.head()
        else:
            return self.table.head(num)

    def tail(self, num: Optional[int] = None) -> pd.DataFrame:
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

    def __repr__(self) -> str:
        """
        Representation of Spectrum object.

        Return
        ------
        the string representation of Spectrum object
        """

        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__repr__()

    def __str__(self) -> str:
        """
        Representation of Spectrum object.

        Return
        ------
        the string representation of Spectrum object
        """
        columns = [column for column in self.features if column in self.table]
        return self.table[columns].__str__()
        

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
            table: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Init ErrorTable object

        Parameters
        ----------
        table : pandas Datarame
            consist error table: error in ppm for mass
        """
        self.table = table

    def md_error_map(
        self, 
        spec: "Spectrum", 
        ppm: float = 3, 
        show_map: Optional[bool] = False
        ) -> pd.DataFrame:
        '''
        Calculate mass differnce map

        Parameters
        ----------
        spec: pd.Dataframe
            Dataframe with spectrum table from Spectrum
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

        df = pd.DataFrame({ 'C':[1,1,1,2,0,1],
                            'H':[2,0,2,1,2,0],
                            'O':[0,1,1,1,1,2]})

        dif_masses = gen_from_brutto(df)['calc_mass'].values
        dif = np.unique([dif_masses*i for i in range(1,10)])

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
        kdm = np.reshape(kernel(positions).T, xx.shape)
        kdm = np.rot90(kdm)

        if show_map:
            fig = plt.figure(figsize=(4,4), dpi=75)
            ax = fig.gca()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.imshow(kdm, extent=[xmin, xmax, ymin, ymax], aspect='auto')
        
        return kdm

    def assign_error(
        self, 
        spec:Spectrum,
        ppm: float = 3,
        brutto_dict = {'C':(4,30), 'H':(4,60), 'O':(0,20)},
        show_map:bool = True):
        '''
        Recallibrate by assign error

        Parameters
        ----------
        spec: Spectrum object
            Initial mass spectrum for recallibrate
        ppm: float
            Permissible relative error in callibrate error. Default 3.
        brutto_dict: dict
            Dictonary with elements ranges for assignment
        show_error: bool
            Optional. Default True. Show process 

        Return
        ------
        ErrorTable object that contain recallabrate error ppm for mass diaposone

        '''
        spectr = copy.deepcopy(spec)
        spectr = spectr.assign(rel_error=ppm, brutto_dict=brutto_dict)
        spectr = spectr.calc_mass().calc_error()

        error_table = spectr.table
        error_table = error_table.loc[:,['mass','rel_error']]
        error_table.columns = ['mass', 'ppm']
        error_table['ppm'] = - error_table['ppm']
        error_table = error_table.dropna()

        kdm = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kdm, 
                            show_map=show_map, 
                            mass=spectr.drop_unassigned().table['mass'].values)
        self = ErrorTable(err).extrapolate((spec.table['mass'].min(), spec.table['mass'].max()))
        self = self.zeroshift(spectr)

        return self

    def massdiff_error(
        self,
        spec: Spectrum,
        show_map:bool = True):
        '''
        Self-recallibration of mass-spectra by mass-difference map

        Parameters
        -----------
        spec: Spectrum object
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
        kdm = self.kernel_density_map(df_error=mde)
        err = self.fit_kernel(f=kdm, show_map=show_map, mass=spec.table['mass'].values)
        
        return ErrorTable(err)

    def etalon_error( self,
                    spec: "Spectrum", #initial masspectr
                    etalon: "Spectrum", #etalon massspectr
                    quart: float = 0.9, #treshold by quartile
                    ppm: float = 3,#treshold by ppm
                    show_map: bool = True
                    ): 
        '''
        Recallibrate by etalon

        Parameters
        ----------
        spec: Spectrum object
            Initial mass spectrum for recallibrate
        etalon: Spectrum object
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

        kdm = self.kernel_density_map(df_error = error_table)
        err = self.fit_kernel(f=kdm, show_map=show_map, mass=spec.table['mass'].values)

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

    def zeroshift(self, spec:"Spectrum") -> "ErrorTable":
        """
        Shift error so mean eror will be zero

        Parameters
        ----------
        spec: Spectrum object
            income massspec

        Return
        ------
        ErrorTable object with shifted ppm error
        """
        err = copy.deepcopy(self)
        mean_error = spec.drop_unassigned().calc_error()['rel_error'].mean()
        err.table['ppm'] = err.table['ppm'] - mean_error
        return ErrorTable(err.table)       


class SpectrumList(UserList):
    """
    Class for work list of Spectrums objects
    inheritan from list class with some extra features.
    Store list of Spectrum objects
    """

    def __init__(self, spectra: Optional[List["Spectrum"]] = []):
        """
        init SpectrumList Class
        
        Parameters
        ----------
        spectra: Sequence[Spectrum]
            list of Spectrum objects
        """
        t = type(Spectrum())
        for spec in spectra:
            if isinstance(spec, t) == False:
                raise Exception(f'SpectrumList must contain only Spectrum objects, not {type(spec)}')

        super().__init__(spectra)
        self.data: List[Spectrum]

    def read_json(self, filename: Union[Path, str]) -> "SpectrumList":
        """
        Read SpectrumList from json, own format

        Parameters
        ----------
        filename: str
            path to SpectrumList json file, absoulute or relative

        Return
        ------
        Spectrum object
        """
        with open(filename, 'rb') as data:
            res = json.load(data)

        for i in res:
            self.append(Spectrum(table = pd.DataFrame(i['table']), metadata=i['metadata']))
        
        return self

    def to_json(self, filename: Union[Path, str]) -> None:
        """
        Saves Spectrum mass-list to JSON own format
        
        Parameters
        ----------
        filename: str
            Path for saving mass spectrum table with calculation to json file
        """

        res = []
        for spec in self:
            out = {'metadata':copy.deepcopy(dict(spec.metadata))}
            out['table'] = spec.table.to_dict()
            res.append(out)

        with open(filename, 'w') as f:
            json.dump(res, f)

    def get_simmilarity(self, mode: str = "cosine", symmetric = True) -> np.ndarray:
        """
        Calculate simmilarity matrix for all spectra in SpectrumList

        Parameters
        ----------
        mode: str
            Optionaly. Default cosine. 
            one of the simmilarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"
        symmetric: bool
            Optionaly. Default True.
            If metric is symmtrical ( a(b)==b(a) ) it is enough to calc just half of table

        Return
        ------
        simmilarity matrix, 2d np.ndarray with size [len(names), len(names)]"""

        
        spec_num = len(self)
        values = np.eye(spec_num)

        for x in range(spec_num):
            if symmetric:
                for y in range(x+1, spec_num):
                    values[x,y] = self[x].simmilarity(self[y], mode=mode)
            else:
                for y in range(spec_num):
                    values[x,y] = self[x].simmilarity(self[y], mode=mode)
        
        if symmetric:
            values = values + values.T - np.diag(np.diag(values))

        return values

    def get_mol_metrics(self, 
                        metrics: Optional[Sequence[str]] = None,
                        how_average: str = 'weight') -> pd.DataFrame:
        """
        Get average metrics

        Parameters
        ----------
        metrics: Sequence[str]
            Optional. Default None. Chose metrics fot watch.
        how_average: str
            How calculate average. My be "count" or "weight" ((default))
        """

        metrics_table = pd.DataFrame()
        names = []

        for i, spec in enumerate(self):
            metr = spec.get_mol_metrics(metrics=metrics, how_average=how_average)
            names.append(spec.metadata['name'])

            if i == 0:
                metrics_table = metr
            else:
                metrics_table = metrics_table.merge(metr, how='outer', on='metric')
                            
        metrics_table.columns = ['metric',*names]

        return metrics_table

    def get_mol_density(self) -> pd.DataFrame:
        """
        Calculate mol class density table

        Return
        ------
        pandas Dataframe with index as mol classes and column as spec name
        """

        mol_density = pd.DataFrame()
        for i, spec in enumerate(self):
            mol_dens_spec = spec.get_mol_class()
            if i == 0:
                index = mol_dens_spec['class'].values
            mol_density[spec.metadata['name']] = mol_dens_spec['density']
        mol_density.index = index

        return mol_density

    def draw_mol_density(
        self,
        mol_density: Optional[pd.DataFrame] = None,
        ax: Optional[plt.axes] = None,
        **kwargs
        ) -> None:
        """
        Draw simmilarity matrix by using seaborn

        Parameters
        ----------
        mol_density: pd.DataFrame
            Optional. Table with molecular class density. Default None and cacl by self.
        ax: matplotlib axes
            Entarnal axes for plot
        **kwargs: dict
            Additional parameters to seaborn heatmap method
        """
        if mol_density is None:
            mol_density = self.get_mol_density()

        if ax is None:
            fig, ax = plt.subplots(figsize=(4,4), dpi=75)

        width=0.35
        
        labels = mol_density.columns
        bottom = np.zeros(len(labels))
        for key in mol_density.index:
            val = [mol_density.at[key, i] for i in labels]
            ax.bar(labels, val, width, label=key, bottom=bottom)
            bottom = bottom + np.array(val)
            
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def draw_simmilarity(
        self,
        mode: str = "cosine",
        values: Optional[np.ndarray] = None,
        ax: Optional[plt.axes] = None,
        annot: bool = True,
        **kwargs
        ) -> None:
        """
        Draw simmilarity matrix by using seaborn

        Parameters
        ----------
        values: np.ndarray
            Optionaly. simmilarity matix.
            Default None - It is call calculate_simmilarity() method.
        mode: str
            Optionaly. If values is none for calculate matrix. 
            Default cosine. one of the simmilarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"
        ax: matplotlib axes
            Entarnal axes for plot
        annotate: bool
            Draw value of simmilarity onto titles
        **kwargs: dict
            Additional parameters to seaborn heatmap method
        """
        if values is None:
            values = self.get_simmilarity(mode=mode)

        if ax is None:
            fig, ax = plt.subplots(figsize=(len(self),len(self)), dpi=75)
        
        axis_labels = []
        for i, spec in enumerate(self):
            axis_labels.append(spec.metadata['name'] if 'name' in spec.metadata else i)
        
        sns.heatmap(np.array(values), vmin=0, vmax=1, cmap="viridis", annot=annot, ax=ax, xticklabels=axis_labels, yticklabels=axis_labels)
        plt.title(mode)


class Tmds(Spectrum):
    """
    A class for calculate TMDS spectrum

    Attributes
    ----------
    spec: MetaData
        Optional. Default None. Metadata object that consist dictonary of metadata
    table: pandas Datarame
        tmds spectrum - mass, intensity and caclulatedd parameters
    metadata: MetaData
        Metadata object that consist dictonary of metadata
    
    """
    def __init__(self, 
                spec: Optional["Spectrum"] = None, 
                table: Optional[pd.DataFrame] = None) -> None:

        if spec is None:
            self.metadata = MetaData()
            self.spec = Spectrum()
        else:
            self.spec = spec.copy()
            self.metadata = spec.metadata
        self.metadata['type'] = 'Tmds'

        if table is None:
            self.table = pd.DataFrame()
        else:
            self.table = table

        super().__init__(self.table, self.metadata)

    def calc(
        self,
        other: Optional["Spectrum"] = None,
        p: float = 0.2,
        wide: int = 10,
        C13_filter:bool = True,
        ) -> "Tmds":

        """
        Total mass difference statistic calculation 

        Parameters
        ----------
        other: Spectrum object
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

        spec = copy.deepcopy(self.spec)
        if other is None:
            spec2 = copy.deepcopy(self.spec)
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

        self.table = tmds_spec

        return self
    
    def calc_by_brutto(
        self,
        ) -> "Tmds":

        """
        Calculate self difference by calculated mass from brutto

        Return
        ------
        Tmds object with assigned signals and elements
        """

        mass = self.spec.drop_unassigned().calc_error().table['calc_mass'].values
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

        self.table = diff_spec

        return self

    def assign(
        self,
        generated_bruttos_table: Optional[pd.DataFrame] = None,
        error: float = 0.001,
        brutto_dict: Optional[dict] = None,
        max_num: Optional[int] = None
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
        
        self.table = res

        return self


if __name__ == '__main__':
    pass