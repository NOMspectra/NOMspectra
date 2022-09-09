from nhsmass.graph import GraphMetric, GraphVis
from nhsmass.spectrum import Spectrum
import pytest
import pandas as pd
import os
import numpy as np

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True).assign().drop_unassigned()
spec1 = spec1.noise_filter(quantile=0.9)

class Test_GraphMetric:

    def test_init(self):
        gm = GraphMetric()
        assert isinstance(gm.spec, type(spec1))
        gm = GraphMetric(spec1)
        assert isinstance(gm.spec, type(spec1))

    def test_get_graph_table(self):
        gm = GraphMetric(spec1)
        df = gm.get_graph_table(length=4)
        test_data = {'nodes':[380, 382, 387, 383, 376],
                     'chains':[75,70,51,61,67],
                     'max_chain':[7,7,12,10,8],
                     'median_chain':[5,5,7,6,6]}
        for key in test_data.keys():
            assert all(a==b for a,b in zip(df[key].to_list(), test_data[key]))

    def test_graph_features(self):
        res = GraphMetric.graph_features(mass=[10,12,24,36,50,62,63], dif=12)
        assert all(a==b for a,b in zip(res, [5,1,3,3]))

class Test_GraphVis:

    @pytest.mark.parametrize('count, result', [([1,2,0], (14.015650, 'CH2')),
                                               ([1,0,1], (27.994915, 'CO')),
                                               ([1,0,2], (43.989830, 'CO2'))])
    def test_gen_diftable(self, count, result):
        df = GraphVis.gen_diftable(el=['C','H','O'], count=[count])
        assert df.loc[0,'calc_mass'] == result[0]
        assert df.loc[0,'names'] == result[1]

    def test_generate(self):
        graph = GraphVis.generate(spec1)
        assert len(graph.G.edges) == 874
        assert len(graph.G.nodes) == 398

    def test_to_gml(self):
        graph = GraphVis.generate(spec1)
        graph.to_gml(os.path.join(root, 'tests', 'temp', 'test_graph.gml'))