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

from typing import Sequence
from .spectrum import Spectrum
from .diff import Tmds
from .brutto import gen_from_brutto

import numpy as np
import pandas as pd
import copy
import networkx as nx
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex


class GraphMetric(object):
    """
    Calculate metric based on graph properties
    """
    def __init__(self, spec:"Spectrum" = None) -> None: 
        if spec is None:
            self.spec = Spectrum()
        else:
            self.spec = spec.copy()

    def significance_fdcel(self, nodes:dict, dif:float) -> float:
        """
        Calculate length parameters of chain Graph with dif mass into mass-list

        Parameters
        ----------
        nodes: dict of nodes
            mass:intensivity. mass rounded by 6
        dif: diff_mass
            diff-mass inm rounded by 6
        
        Return
        ------
        tuple (0,1)
        [0]:float - sum of n/total nodes
        where n - length of chain, 
        [1]:float - sum of all intensity includes in node 
        """

        G = nx.DiGraph()
        
        mass = np.array(list(nodes.keys()))
        intensity = np.array(list(nodes.values()))

        for item in mass:
            res = np.round(item + dif,6)
            if res in mass:
                G.add_edge(item, res)

        roots = [n for (n, d) in G.in_degree if d == 0]
        leafs = [n for (n, d) in G.out_degree if d == 0]

        total_intens = np.sum(intensity[np.isin(mass, list(G.nodes))])
        branches = []

        for root in roots:
            path = list(nx.algorithms.all_simple_paths(G, root, leafs))
            n = len(path[0])
            s = 0
            for item in path[0]:
                s = s + nodes[item]
            branches.append(n*s/total_intens)

        return np.sum(branches), total_intens

    def get_FDCEL_table(self, length:int = 50) -> pd.DataFrame:
        """
        Calculate FDCEL (formulae differences chains Expected Length)

        Parameters
        ----------
        spec: MassSpectrm
            income mass spectrum with assigned brutti formulas
        length: int
            length of out fdcel vector

        Return
        ------
        Pandas Dataframe with columns 'calc_mass', 'significance', 'difsum_intens'
        """

        spec = self.spec.copy().drop_unassigned().calc_mass()
        diff = Tmds(spec).calc_by_brutto().assign(max_num=length).calc_mass()
        mass = spec.table['calc_mass'].values
        intensivity = spec.table['intensity'].values
        
        nodes = dict(zip(mass, intensivity))
        diff.table['significance'] = 0
        diff.table['difsum_intens'] = 0
        
        tim = len(diff.table)
        for i, row in tqdm(diff.table.iterrows(), total=tim):
            dif = row['calc_mass']
            gr, nd = self.significance_fdcel(nodes, dif)
            diff.table.loc[i, 'significance'] = gr
            diff.table.loc[i, 'difsum_intens'] = nd

        return diff.table.loc[:,['calc_mass', 'significance', 'difsum_intens']]

    @staticmethod
    def calc_fdcel(spec1: "Spectrum", spec2: "Spectrum") -> float:
        """
        Calculate FDCEL

        Parameters
        ----------
        spec1: MassSpectrum object
            first spec
        spec2: MassSpectrum object
            second spec

        Return
        ------
        float: FDCEL value
        """

        fdecel_table1 = GraphMetric(spec1).get_FDCEL_table()
        fdecel_table2 = GraphMetric(spec2).get_FDCEL_table()

        res = fdecel_table1.merge(fdecel_table2, on='calc_mass')

        total_intens = spec1.drop_unassigned().table['intensity'].sum() + spec2.drop_unassigned().table['intensity'].sum()
        res['w'] = (res['difsum_intens_x'] + res['difsum_intens_y'])/total_intens
        res['fdcel'] = res['w'] * np.fabs(res['significance_x'] - res['significance_y'])
        fdcel = res['fdcel'].sum()/len(res)

        return fdcel

    def graph_features(self, mass:np.array, dif:float) -> float:
        """
        Calculate chain characteristic for diff mass

        Parameters
        ----------
        mass:
            mass:intensivity. mass rounded by 6
        dif: diff_mass
            diff-mass inm rounded by 6
        
        Return
        ------
        tuple of characteristics
        'nodes', 'chains', 'max_chain', 'median chain'
        """

        G = nx.DiGraph()
        
        for item in mass:
            res = np.round(item + dif,6)
            if res in mass:
                G.add_edge(item, res)
        
        roots = [n for (n, d) in G.in_degree if d == 0]
        leafs = [n for (n, d) in G.out_degree if d == 0]

        branches = []

        for root in roots:
            path = list(nx.algorithms.all_simple_paths(G, root, leafs))
            n = len(path[0])
            if n > 2:
                branches.append(n)

        return len(list(G.nodes)), len(branches), np.max(branches), np.median(branches)

    def get_graph_table(self, length:int = 50) -> pd.DataFrame:
        """
        Calculate chains characteristic in graph for mass diff

        Parameters
        ----------
        spec: MassSpectrm
            income mass spectrum with assigned brutti formulas
        length: int
            length of out fdcel vector

        Return
        ------
        Pandas Table with characteristics
        Add to TMDS table 'nodes', 'chains', 'max_chain', 'median chain'
        """

        spec = self.spec.copy().drop_unassigned().calc_mass()
        diff = Tmds(spec).calc_by_brutto().assign(max_num=length).calc_mass()
        mass = spec.table['calc_mass'].values
        
        tim = len(diff.table)
        res = []
        for i, row in tqdm(diff.table.iterrows(), total=tim):
            dif = row['calc_mass']
            res.append(self.graph_features(mass, dif))

        feat = ['nodes', 'chains', 'max_chain', 'median chain']
        diff.table = diff.table.join(pd.DataFrame(data = res, columns=feat))

        return diff.table

class Vis(object):
    """
    Generate visulization
    """

    def __init__(self, G:nx.Graph=None) -> None:
        self.G = G

    @staticmethod
    def generate(spec:"Spectrum", 
                dif_table:pd.DataFrame = None,
                metrics: Sequence[str] = None
                ) -> 'Vis':
        """
        Generate direct grpah from massspectrum and difference table

        Parameters
        ----------
        spec: MassSpectrum object
            mass spec with assigned brutto
        dif_table: pd.DataFrame
            table contatin columns 'calc_mass' and optional 'name', 'color'.
            Optional. if None - generate default dif table.
        metric: Sequence[str]
            Metrics in Spectrum that wil be included to metadata of graph
        """
        spec = spec.copy()
        spec = spec.drop_unassigned().calc_all_metrics()
        mass = spec.table['calc_mass'].values

        if dif_table is None:
            dif_table = Vis.gen_diftable()

        if metrics is None:
            metrics = spec.table.columns
        
        m = {}

        G = nx.DiGraph()

        for i, dif_row in dif_table.iterrows():
            dif = dif_row['calc_mass']

            if 'names' in dif_row:
                dif_name = dif_row['names']
            else:
                dif_name = str(dif)
            if 'color' in dif_row:
                dif_color = dif_row['color']
            else:
                dif_color = '#07438c'

            for item in mass:
                res = np.round(item + dif,6)
                if res in mass:
                    atr_i = {metric.replace('/','').replace('-',''):str(spec.table.loc[spec.table['calc_mass']==item, metric].values[0]) for metric in metrics}
                    atr_r = {metric.replace('/','').replace('-',''):str(spec.table.loc[spec.table['calc_mass']==res, metric].values[0]) for metric in metrics}
                    G.add_node(str(item), **atr_i)
                    G.add_node(str(res), **atr_r)
                    G.add_edge(str(item), str(res), weight=dif, name=dif_name, color=dif_color)

        return Vis(G)
    
    @staticmethod
    def gen_diftable(el = None, count = None, colors=True) -> pd.DataFrame:
        """
        Generate dif table for plotting graph

        Parameters
        ----------
        col: list of str
            Optional. list with elements.
            Default = ['C','H','O']
        count: list with list
            Optional. Count of elements.
            Default
            [[1,2,0],
            [1,0,1],
            [1,0,2]]
        colors: bool
            if colors generate colors for difmasses

        Return
        ------
        pd.DataFrame with most usual diffmass        
        """

        if el is None:
            el = ['C','H','O']
        if count is None:
            count = [[1,2,0],
                    [1,0,1],
                    [1,0,2]]

        dif_table = pd.DataFrame(data=count, columns=el)
        dif_table = gen_from_brutto(dif_table)
        
        col = []
        for row in count:
            c = ''
            for i, e in enumerate(el):
                if row[i] == 1:
                    c = c + f'{e}'
                elif row[i] != 0:
                    c = c + f'{e}{int(row[i])}'
            col.append(c)
        dif_table['names'] = col

        if colors:
            col = []
            cmap = cm.get_cmap('hsv', len(count)+1)   
            for i in range(cmap.N):
                rgba = cmap(i)
                col.append(rgb2hex(rgba))
            dif_table['color'] = col[1:]

        return dif_table

    def to_gml(self, filename:str) -> None :
        """
        Generate gml file, that can be open in other programm such as Cytosacpe

        Parameters
        ----------
        filename: str
            file to save html
        """
        nx.write_gml(self.G, filename)


if __name__ == '__main__':
    pass