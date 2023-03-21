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

import os
from pathlib import Path
from typing import List, Sequence, Union, Optional
import copy
import json
from collections import UserList
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .spectrum import Spectrum


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

    @staticmethod
    def read_json(filename: Union[Path, str]) -> "SpectrumList":
        """
        Read SpectrumList from json

        Parameters
        ----------
        filename: str
            path to SpectrumList json file, absoulute or relative

        Return
        ------
        SpectrumList
        """

        specs = SpectrumList()

        with open(filename, 'rb') as data:
            res = json.load(data)

        for i in res:
            specs.append(Spectrum(table = pd.DataFrame(i['table']), metadata=i['metadata']))
        
        return specs

    def to_json(self, filename: Union[Path, str]) -> None:
        """
        Saves Spectrum mass-list to JSON
        
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

    def to_csv(self, folder: Union[Path, str]) -> None:
        """
        Save all spectrum into separate csv files

        Parameters
        ----------
        folder: str
            folder for save spectra in separate files
        """

        for spec in self:
            name = f'{spec.metadata["name"]}.csv'
            spec.table.to_csv(os.path.join(folder, name))

    @staticmethod
    def read_csv(folder: Union[Path, str]) -> 'SpectrumList':
        """
        Read csv files from folder to SpectrumList object. 
        Read only 'csv' ot 'txt' fromat

        Parameters
        ----------
        folder: str
            folder for save spectra in separate files

        Return
        ------
        SpectrumList
        """

        specs = SpectrumList()

        for file in os.listdir(folder):
            if file[-3:].lower() == 'txt' or file[-3:].lower() == 'csv':
                spec = Spectrum.read_csv(filename=os.path.join(folder, file))
                specs.append(spec)
        
        return specs

    def get_names(self) -> Sequence:
        """
        Get names of spectra

        Return
        ------
        List
            list with names of spectra in SpectrumList
        """
        return [spec.metadata['name'] for spec in self]

    def get_simmilarity(self, mode: str = "cosine", symmetric = True) -> np.ndarray:
        """
        Calculate simmilarity matrix for all spectra in SpectrumList

        Parameters
        ----------
        mode: {"tanimoto", "jaccard", "cosine"}
            Optionaly. Default cosine. 
            one of the simmilarity functions
            Mode can be: "tanimoto", "jaccard", "cosine"
        symmetric: bool
            Optionaly. Default True.
            If metric is symmetrical ( a(b)==b(a) ) it is enough to calc just half of table

        Return
        ------
        numpy array
            table with simmilirities of spectrum corresponig their index in SpectrumList
        """
        
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
                        func: Optional[str] = None) -> pd.DataFrame:
        """
        Get average molecular metrics

        Parameters
        ----------
        metrics: Sequence[str]
            Optional. Default None. Chose metrics fot watch.
        func: {"weight", "mean", "median", "max", "min", "std"}
            How calculate average. My be "weight" (default - weight average on intensity),
            "mean", "median", "max", "min", "std" (standard deviation)

        Return
        ------
        Pandas Dataframe 
        """

        metrics_table = pd.DataFrame()        
        names = []

        for i, spec in enumerate(self):
            metr = spec.get_mol_metrics(metrics=metrics, func=func)
            names.append(spec.metadata['name'])
            if i == 0:
                index = metr['metric'].values
            metrics_table[spec.metadata['name']] = metr['value']

        metrics_table.index = index
        metrics_table.columns = names

        return metrics_table

    def get_square_vk(self, how_average: str = 'weight') -> pd.DataFrame:
        """
        Calculate Van-Krevelen square density for spectra

        Parameters
        ----------
        how_average: {"count", "weight"}
            How calculate average. My be "count" or "weight" ((default))

        Return
        ------
        Pandas Dataframe

        References
        ----------
        Perminova I. V. From green chemistry and nature-like technologies towards 
        ecoadaptive chemistry and technology // Pure and Applied Chemistry. 
        2019. Vol. 91, № 5. P. 851-864.
        """

        square_vk = pd.DataFrame()
        for i, spec in enumerate(self):
            square_dens = spec.get_squares_vk(how_average=how_average)
            if i == 0:
                index = square_dens['square'].values
            square_vk[spec.metadata['name']] = square_dens['value']
        square_vk.index = index

        return square_vk

    def get_mol_density(self, how_average: str = "weight", how: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate molecular class density table

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

        mol_density = pd.DataFrame()
        for i, spec in enumerate(self):
            mol_dens_spec = spec.get_mol_class(how_average=how_average, how=how)
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
        Draw mol density of spectra in bar diagram

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

        ax.set_xticklabels(labels, rotation=90, ha='right')
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5))

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


if __name__ == '__main__':
    pass