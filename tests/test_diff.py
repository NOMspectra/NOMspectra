from natorgms.diff import Tmds, assign_by_tmds
from natorgms.spectrum import Spectrum
import pytest
import os
import numpy as np
import pandas as pd

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True).assign()
spec1 = spec1.noise_filter(quantile=0.8)

df_test = pd.DataFrame({'mass':[12, 24, 36, 48, 60, 72, 84, 96], 
                        'intensity':[1,1,1,1,1,1,1,1],
                        'C':[1,2, np.NaN, 4, np.NaN,6,7,8],
                        'N':[0,0,0,0,0,0,0,0],
                        'H':[0,0,0,0,0,0,0,0],
                        'O':[0,0,0,0,0,0,0,0],
                        'assign':[True, True, False, True, False, True, True, True]})

class Test_assign_by_tmds():

    def test_assign_by_tmds(self):
        spec = assign_by_tmds(spec1).drop_unassigned()
        assert len(spec) == 1859

    def test_max_num(self):
        spec = assign_by_tmds(spec1, max_num=5).drop_unassigned()
        assert len(spec) == 1828
    
    def test_pvalue(self):
        spec = assign_by_tmds(spec1, p=1).drop_unassigned()
        assert len(spec) == 1844

    def test_c13(self):
        spec = spec1.noise_filter(quantile=0.5)
        spec = assign_by_tmds(spec, C13_filter=False).drop_unassigned()
        assert len(spec) == 1117

class Test_tmds:

    def test_init(self):
        tmds = Tmds()

    def test_calc(self):
        spec = Spectrum(df_test)
        tmds = Tmds(spec).calc(C13_filter=False)
        assert tmds.table.loc[0,'mass'] == 12
        assert round(tmds.table.loc[0,'intensity'],3) == 2.333

    def test_calc_by_brutto(self):
        spec = Spectrum(df_test)
        tmds = Tmds(spec).calc_by_brutto()
        assert tmds.table.loc[1,'mass'] == 12
        assert tmds.table.loc[1,'intensity'] == 0.5
        assert tmds.table.loc[3,'mass'] == 36
        assert round(tmds.table.loc[3,'intensity'],3) == 0.333

    def test_assign(self):
        spec = Spectrum(df_test)
        tmds = Tmds(spec).calc_by_brutto()
        tmds = tmds.assign()
        res = tmds.table['C'].to_list()
        assert all(a==b for a,b in zip(res, [i for i in range(1,8)]))