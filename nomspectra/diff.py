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

from typing import Dict, Optional
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

from .spectrum import Spectrum
from .brutto import brutto_gen
from .metadata import MetaData


def assign_by_tmds(
    spec: "Spectrum", 
    tmds_spec: Optional["Tmds"] = None,
    tmds_brutto_dict: Optional[Dict] = None, 
    rel_error: float = 3,
    p = 0.2,
    max_num: Optional[int] = None,
    C13_filter: bool = True
    ) -> "Spectrum":
    '''
    Assigne brutto formulas by TMDS

    Additianal assignment of masses that can't be done with usual methods

    Parameters
    ----------
    spec: Spectrum object
        Mass spectrum for assign by tmds
    tmds_spec: Tmds object
        Optional. if None generate tmds spectr with default parameters
        Tmds object, include table with most intensity mass difference
    brutto_dict: dict
        Optional. Deafault None.
        Custom Dictonary for generate brutto table.
        Example: {'C':(-1,20),'H':(-4,40), 'O':(-1,20),'N':(-1,2)}
    abs_error: float
        Optional, default 1 ppm. Relative error for assign peaks by massdif
    p: float
        Optional. Default 0.2. 
        Relative intensity coefficient for treshold tmds spectrum
    max_num: int
        Optional. Max mass diff numbers
    C13_filter: bool
        Use only peaks with C13 isotope peak for generate tmds. Default True.

    Return
    ------
    Spectrum
        Assigned by tmds masses

    Reference
    ---------
    Kunenkov, Erast V., et al. "Total mass difference 
    statistics algorithm: a new approach to identification 
    of high-mass building blocks in electrospray ionization 
    Fourier transform ion cyclotron mass spectrometry data 
    of natural organic matter." 
    Analytical chemistry 81.24 (2009): 10106-10115.
    '''
    
    if "assign" not in spec.table:
        raise Exception("Spectrum is not assigned")

    spec = spec.copy()

    #calculstae tmds table
    if tmds_spec is None:
        tmds_spec = Tmds(spec=spec).calc(p=p, C13_filter=C13_filter) #by varifiy p-value we can choose how much mass-diff we will take
        tmds_spec = tmds_spec.assign(max_num=max_num, brutto_dict=tmds_brutto_dict)
        tmds_spec = tmds_spec.calc_mass()

    #prepare tmds table
    tmds = tmds_spec.table.sort_values(by='intensity', ascending=False).reset_index(drop=True)
    tmds = tmds.loc[tmds['intensity'] > p].sort_values(by='mass', ascending=True).reset_index(drop=True)
    elem_tmds = tmds_spec.find_elements()

    #prepare spec table
    assign_false = copy.deepcopy(spec.table.loc[spec.table['assign'] == False]).reset_index(drop=True)
    assign_true = copy.deepcopy(spec.table.loc[spec.table['assign'] == True]).reset_index(drop=True)
    masses = assign_true['mass'].values
    elem_spec = spec.find_elements()
    
    
    #Check that all elements in tmds also in spec
    if len(set(elem_tmds)-set(elem_spec)) > 0:
        raise Exception(f"All elements in tmds spectrum must be in regular spectrum too. But {(set(elem_tmds)-set(elem_spec))} not in spectrum")
    for i in set(elem_spec)-set(elem_tmds):
        tmds[i] = 0

    mass_dif_num = len(tmds)
    min_mass = np.min(masses)

    for i, row_tmds in tqdm(tmds.iterrows(), total=mass_dif_num):

        mass_shift = - row_tmds['calc_mass']
        
        for index, row in assign_false.iterrows():
            if row['assign'] == True:
                continue
                    
            mass = row["mass"] + mass_shift
            if mass < min_mass:
                continue

            idx = np.searchsorted(masses, mass, side='left')
            if idx > 0 and (idx == len(masses) or np.fabs(mass - masses[idx - 1]) < np.fabs(mass - masses[idx])):
                idx -= 1
                
            if np.fabs(masses[idx] - mass) / mass * 1e6 <= rel_error:
                assign_false.loc[index,'assign'] = True

                for el in elem_spec:
                    assign_false.loc[index,el] = row_tmds[el] + assign_true.loc[idx,el]

    assign_true = pd.concat([assign_true, assign_false], ignore_index=True).sort_values(by='mass').reset_index(drop=True)
    
    out = Spectrum(assign_true)
    out = out.calc_mass()

    out.table=out.table[out.table['calc_mass'].isnull() | ~out.table[out.table['calc_mass'].notnull()].duplicated(subset='calc_mass',keep='first')] 
    spec.table = out.table.sort_values(by='mass').reset_index(drop=True)

    spec.metadata.add({'assigned_by_tmds':True})
    
    return spec


class Tmds(Spectrum):
    """
    A class for calculate TMDS (Total mass difference 
    statistics) spectrum

    Reference
    ---------
    Kunenkov, Erast V., et al. "Total mass difference 
    statistics algorithm: a new approach to identification 
    of high-mass building blocks in electrospray ionization 
    Fourier transform ion cyclotron mass spectrometry data 
    of natural organic matter." 
    Analytical chemistry 81.24 (2009): 10106-10115.
    """

    def __init__(self, 
                spec: Optional["Spectrum"] = None, 
                table: Optional[pd.DataFrame] = None) -> None:
        """
        Parameters
        ----------
        table: pandas Datarame
            tmds spectrum - mass, intensity and caclulatedd parameters
        metadata: MetaData
            Metadata object that consist dictonary of metadata
        """

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
            Minimum relative intensity for taking mass-difference. Default 0.2.
        wide: int
            Minimum interval in 0.001*wide Da of peaks finding. Default 10.
        C13_filter: bool
            Use only peaks that have C13 isotope peak. Default True

        Return
        ------
        Tmds
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
            raise Exception(f"Too low number of assigned peaks")

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
    
    def calc_by_brutto(self) -> "Tmds":

        """
        Calculate self difference by calculated mass from brutto

        Return
        ------
        Tmds
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
        Tmds
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