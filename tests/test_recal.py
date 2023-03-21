from nomspectra.recal import recallibrate, ErrorTable
from nomspectra.spectrum import Spectrum
import pytest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
sample2_path = os.path.join(root, 'tests', 'sample2.txt')
spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True).assign().drop_unassigned()
mapper = {'m/z':'mass','I':'intensity'}
spec2 = Spectrum.read_csv(sample2_path, mapper=mapper, take_only_mz=True, sep='\t').assign().drop_unassigned()

@pytest.mark.parametrize('how, result', [("assign", 0.016),
                                            ("mdm", -0.006)])
def test_recallibrate(how, result):
    spec = recallibrate(spec2, how=how)
    spec = spec.calc_error()
    assert round(spec.table['rel_error'].mean(),3) == result

def test_recallibrate_by_etalon():
    spec = recallibrate(spec2, how=sample1_path)
    spec = spec.calc_error()
    assert round(spec.table['rel_error'].mean(),3) == 0.053

class Test_ErrorTable:

    def test_init(self):
        et = ErrorTable()
    
    def test_md_error_map(self):
        et = ErrorTable.md_error_map(spec1)
        assert len(et) == 26309

    def test_md_error_map_ppm(self):
        et = ErrorTable.md_error_map(spec1, ppm=0.5)
        assert len(et) == 25916
    
    def test_kernel_density_map(self):
        et = ErrorTable.md_error_map(spec1)
        f = ErrorTable.kernel_density_map(et)
        assert len(f) == 100
        assert round(np.mean(f[50,:]),3) == 0.011

    def test_kernel_density_map_ppm(self):
        et = ErrorTable.md_error_map(spec1)
        f = ErrorTable.kernel_density_map(et, ppm=0.5)
        assert len(f) == 100
        assert round(np.mean(f[50,:]),3) == 0.013

    def test_fit_kernel(self):
        et = ErrorTable.md_error_map(spec1)
        f = ErrorTable.kernel_density_map(et)
        fit = ErrorTable.fit_kernel(f, mass=spec1.table['mass'].values, show_map=True)
        assert round(fit['ppm'].mean(),3) == 0.012

    def test_fit_kernel(self):
        et = ErrorTable.massdiff_error(spec1)
        assert round(et.table['ppm'].mean(),6) == 0.012114

    def test_fit_kernel(self):
        et = ErrorTable.assign_error(spec1)
        assert round(et.table['ppm'].mean(),6) == -0.027075

    def test_fit_kernel(self):
        et = ErrorTable.etalon_error(spec1, spec2)
        assert round(et.table['ppm'].mean(),6) == -0.069091

    def test_extrapolate(self):
        df = pd.DataFrame({'mass':[400,500],'ppm':[0,1]})
        et = ErrorTable(df)
        et = et.extrapolate((300, 700))
        assert et.table.loc[et.table['mass']==300, 'ppm'].values[0] == -1
        assert et.table.loc[et.table['mass']==700, 'ppm'].values[0] == 3

    def test_show_error(self):
        et = ErrorTable.assign_error(spec1)
        et.show_error()