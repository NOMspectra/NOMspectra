#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nhsmasslib. 
#    Developed in Natural Humic System laboratory
#    Moscow State University (Head of lab - Perminova I.V.)
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

from .mass import MassSpectrum, Tmds
from .brutto import gen_from_brutto

import numpy as np
import pandas as pd
import copy
import networkx as nx
from pyvis.network import Network
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex


class Metric(object):
    """
    Calculate metric based on graph properties

    Atributes
    ---------
    spec: MassSpectrum object
    Massspectrum for calculate
    """
    def __init__(self, spec:"MassSpectrum" = None) -> None: 
        self.spec = spec

    def significance_fdcel(self, nodes:dict, dif:float) -> float:
        """
        Calculate length parameters of chain Graph with dif mass into mass-list

        Parameters
        ----------
        nodes: dict of nodes
            mass:intensivity. mass rounded by 6
        dif: diff_mass
            diff-mass inm rounded by 6
        
        Retun
        -----
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
        Pandas Dataframe with columns 'calculated_mass', 'significance', 'difsum_intens'
        """

        spec = copy.deepcopy(self.spec).drop_unassigned().calculate_mass()
        diff = Tmds().calc_by_brutto(spec).assign(max_num=length).calculate_mass()
        mass = spec.table['calculated_mass'].values
        intensivity = spec.table['intensity'].values
        
        nodes = dict(zip(mass, intensivity))
        diff.table['significance'] = 0
        diff.table['difsum_intens'] = 0
        
        tim = len(diff.table)
        for i, row in tqdm(diff.table.iterrows(), total=tim):
            dif = row['calculated_mass']
            gr, nd = self.significance_fdcel(nodes, dif)
            diff.table.loc[i, 'significance'] = gr
            diff.table.loc[i, 'difsum_intens'] = nd

        return diff.table.loc[:,['calculated_mass', 'significance', 'difsum_intens']]

    def calc_fdcel(self, spec1: "MassSpectrum", spec2: "MassSpectrum") -> float:
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

        fdecel_table1 = Metric(spec1).get_FDCEL_table()
        fdecel_table2 = Metric(spec2).get_FDCEL_table()

        res = fdecel_table1.merge(fdecel_table2, on='calculated_mass')

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
        
        Retun
        -----
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

        spec = copy.deepcopy(self.spec).drop_unassigned().calculate_mass()
        diff = Tmds().calc_by_brutto(spec).assign(max_num=length).calculate_mass()
        mass = spec.table['calculated_mass'].values
        
        tim = len(diff.table)
        res = []
        for i, row in tqdm(diff.table.iterrows(), total=tim):
            dif = row['calculated_mass']
            res.append(self.graph_features(mass, dif))

        feat = ['nodes', 'chains', 'max_chain', 'median chain']
        diff.table = diff.table.join(pd.DataFrame(data = res, columns=feat))

        return diff.table

class Vis(object):
    """
    Generate visulization

    Atributes
    ---------
    Networkx Graph
    """

    def __init__(self, G:nx.Graph=None) -> None:
        self.G = G

    def generate(self, 
                spec:"MassSpectrum", 
                dif_table:pd.DataFrame = None,
                brutto_name = True
                ) -> nx.DiGraph:
        """
        Generate direct grpah from massspectrum and difference table

        Parameters
        ----------
        spec: MassSpectrum object
            mass spec with assigned brutto
        dif_table: pd.DataFrame
            table contatin columns 'calculated_mass' and optional 'name', 'color'.
            Optional. if None - generate default dif table.
        brutto_name: bool
            Optional. Default True. Replace mass for brutto in graph
        """
        spec = copy.deepcopy(spec)
        spec = spec.drop_unassigned().calculate_mass()
        mass = spec.table['calculated_mass'].values

        if dif_table is None:
            dif_table = self.gen_diftable()

        if brutto_name:
            spec = spec.calc_brutto()
            brutto = spec.table['brutto'].to_list()
            massl = spec.table['calculated_mass'].to_list()
            nods = dict(zip(massl, brutto))
        
        G = nx.Graph()

        for i, dif_row in dif_table.iterrows():
            dif = dif_row['calculated_mass']

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
                    if brutto_name:
                        G.add_node(nods[item])
                        G.add_node(nods[res])
                        G.add_edge(nods[item], nods[res], weight=dif_name, color=dif_color)
                    else:
                        G.add_node(str(item))
                        G.add_node(str(res))
                        G.add_edge(str(item), str(res), weight=dif_name, color=dif_color)
        return Vis(G)

    def to_html(self, filename:str, size:str ='800px') -> None :
        """
        Generate html with graph

        filename: str
            file to save html
        size: str
            optional, default 800px. Size of graph        
        """

        net = Network(size, size)
        net.from_nx(self.G)
        net.show_buttons(filter_=['physics'])
        net.show(f'{filename}')

    def gen_diftable(self, el = None, count = None, colors=True) -> pd.DataFrame:
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
            [0,2,0],
            [1,2,1],
            [1,0,2],
            [0,2,1]]
        colors: bool
            if colors generate colors for difmasses

        Return
        ------
        pd.DataFrame with most usual diffmass        
        """
        default = False

        if el is None:
            el = ['C','H','O']
        if count is None:
            default = True
            count = [[1,2,0],
                    [1,0,1],
                    [0,2,0],
                    [1,2,1],
                    [1,0,2],
                    [0,2,1]]

        dif_table = pd.DataFrame(data=count, columns=el)
        dif_table = gen_from_brutto(dif_table)
        
        col = []
        for row in count:
            c = ''
            for i, e in enumerate(el):
                if row[i] == 1:
                    c = c + f'{e}'
                elif row[i] > 0:
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


if __name__ == '__main__':
    pass