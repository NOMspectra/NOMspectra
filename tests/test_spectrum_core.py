from nomhsms.spectrum import Spectrum
import pytest
import os
import numpy as np
import pandas as pd

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
sample2_path = os.path.join(root, 'tests', 'sample2.txt')

class Test_load_and_save:

    def test_read_csv(self):
        spec = Spectrum.read_csv(sample1_path)  
        all_columns = set(['mass', 'intensity', 'C', 'O', 'H', 'N', 'C_13','assign','calc_mass', 'abs_error', 'rel_error'])
        assert all_columns == set(spec.table.columns)
        assert len(spec.table) == 13374
        assert spec.metadata['name'] == 'sample1'

    def test_ignore_columns(self):
        ignore = ['abs_error', 'rel_error']
        spec = Spectrum.read_csv(sample1_path, ignore_columns=ignore)
        all_columns = set(['mass', 'intensity', 'C', 'O', 'H', 'N', 'C_13', 'assign','calc_mass'])
        assert all_columns == set(spec.table.columns)

    def test_take_columns(self):
        take = ['mass', 'intensity', 'C', 'H', 'N', 'O', 'C_13']
        spec = Spectrum.read_csv(sample1_path, take_columns=take)        
        all_columns = set(['mass', 'intensity', 'C', 'O', 'H', 'N', 'C_13'])
        assert all_columns == set(spec.table.columns)

    def test_take_only_mz(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        all_columns = set(['mass', 'intensity'])
        assert all_columns == set(spec.table.columns)

    def test_mapper_sep(self):
        mapper = {'m/z':'mass','I':'intensity'}
        spec = Spectrum.read_csv(sample2_path, mapper=mapper, take_only_mz=True, sep='\t')
        all_columns = set(['mass', 'intensity'])
        assert all_columns == set(spec.table.columns)

    def test_mass_min(self):
        spec = Spectrum.read_csv(sample1_path, mass_min=300)
        assert spec.table['mass'].min() > 300

    def test_mass_max(self):
        spec = Spectrum.read_csv(sample1_path, mass_max=600)
        assert spec.table['mass'].min() < 600

    def test_intensity_min(self):
        spec = Spectrum.read_csv(sample1_path, intens_min=100)
        assert spec.table['intensity'].min() > 100

    def test_intensity_max(self):
        spec = Spectrum.read_csv(sample1_path, intens_max=100000000)
        assert spec.table['intensity'].max() < 100000000

    def test_assign_mark(self):
        spec = Spectrum.read_csv(sample1_path, assign_mark=True)  
        assert 'assign' in spec.table.columns

    def test_to_csv(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        test_path = os.path.join(root, 'tests', 'test.csv')
        spec.to_csv(test_path)
        spec2 = Spectrum.read_csv(test_path)
        os.remove(test_path)
        for col in spec.table.columns:
            assert all(a==b for a, b in zip(spec.table[col].round(6).to_list(), spec2.table[col].round(6).to_list()))
            
    def test_json(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        test_path = os.path.join(root, 'tests', 'test.json')
        spec.to_json(test_path)
        spec2 = Spectrum.read_json(test_path)
        os.remove(test_path)
        assert spec.metadata == spec2.metadata
        for col in spec.table.columns:
            assert all(a==b for a, b in zip(spec.table[col].round(6).to_list(), spec2.table[col].round(6).to_list()))
    
def test_find_elements():
    spec = Spectrum.read_csv(sample1_path)
    elems = spec.find_elements()
    test = ['C','H','O','N','C_13']
    assert all(a==b for a,b in zip(test, elems))

class Test_assign:

    def test_assign(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        spec = spec.assign()
        assert len(spec.table.loc[spec.table['assign'] == True]) == 4023
        assert spec.metadata['sign'] == '-'
        elems = spec.find_elements()
        test = ['C','H','O','N','S']
        assert all(a==b for a,b in zip(test, elems))

    @pytest.mark.parametrize('param', [('0', 517.122029), ('-', 516.114752), ('+', 517.121480)])
    def test_mode(self, param):
        sign, mass = param
        spec = Spectrum()
        spec.table['mass'] = [mass]
        spec.table['intensity'] = [1]
        spec = spec.assign(sign=sign)
        # mass, intensity, C, H, O, N, S, assign
        test_val = [mass, 1, 24, 23, 12, 1, 0, True]
        val = spec.table.loc[0,:].to_list()
        assert all(a==b for a,b in zip(test_val, val))
        assert spec.metadata['sign'] == sign

    def test_ppm_and_dict(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        brutto_dict = {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'C_13':(0,6)}
        spec = spec.assign(brutto_dict=brutto_dict, rel_error=0.25)
        assert len(spec.table.loc[spec.table['assign'] == True]) == 4857
        elems = spec.find_elements()
        test = ['C','H','O','N','C_13']
        assert all(a==b for a,b in zip(test, elems))

    def test_restrict(self):
        spec = Spectrum.read_csv(sample1_path, take_only_mz=True)
        spec = spec.assign(mass_max=400, mass_min=200, intensity_max=100000000, intensity_min=10000000)
        assert len(spec.table.loc[spec.table['assign'] == True]) == 348

class Test_noise_filter:

    def test_noise_filter(self):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.noise_filter()
        assert len(spec) == 7856

    def test_intensity(self):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.noise_filter(intensity=1000000)
        assert len(spec) == 7010

    def test_force(self):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.noise_filter(force=3)
        assert len(spec) == 4223

    @pytest.mark.parametrize('q, l', [(0.1, 12036), (0.5, 6687), (0.75, 3344)])
    def test_quantile(self, q, l):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.noise_filter(quantile=q)
        assert len(spec) == l

def test_drop_unassigned():
    spec = Spectrum.read_csv(sample1_path)
    spec = spec.drop_unassigned()
    assert len(spec) == 4857
    assert False not in spec.table['assign']
    assert np.NaN not in spec.table['C']

class Test_filter_by_C13:

    def test_filter_by_C13(self):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.filter_by_C13()
        assert len(spec.table.loc[spec.table['C13_peak'] == True]) == 1813

    @pytest.mark.parametrize('test, result', [([199.0, 200.0, 200.003, 201.003355, 202.0], [False, True, False, False, False])])
    def test_values(self, test, result):
        spec = Spectrum()
        spec.table['mass'] = test
        spec.table['intensity'] = [1 for i in test]
        spec = spec.filter_by_C13()
        assert all(a==b for a,b in zip(result, spec.table['C13_peak'].to_list()))

    @pytest.mark.parametrize('ppm, test, result', [(0, [200.0, 201.0033], [False, False]), 
                                                    (0.2, [200.0, 201.0033], [False, False]), 
                                                    (0.5, [200.0, 201.0033], [True, False])])
    def test_ppm(self, ppm, test, result):
        spec = Spectrum()
        spec.table['mass'] = test
        spec.table['intensity'] = [1 for i in test]
        spec = spec.filter_by_C13(rel_error=ppm)
        assert all(a==b for a,b in zip(result, spec.table['C13_peak'].to_list()))

    def test_remove(self):
        spec = Spectrum.read_csv(sample1_path)
        spec = spec.filter_by_C13(remove=True)
        assert len(spec) == 1571


@pytest.mark.parametrize('how, result', [('sum', 0.004),
                                            ('max', 1),
                                            ('mean', 50.676),
                                            ('median', 147.987)])
def test_normalize(how, result):
    spec = Spectrum.read_csv(sample1_path)
    spec = spec.normalize(how=how)
    assert round(spec.table['intensity'].max(), 3) == result

def test_merge_isotopes():
    spec = Spectrum.read_csv(sample1_path).drop_unassigned()
    values = (spec.table['C'] + spec.table['C_13']).copy().to_list()
    spec = spec.merge_isotopes()
    assert 'C_13' not in spec.table.columns
    assert all(a==b for a,b in zip(values, spec.table['C'].to_list()))

def test_copy():
    spec = Spectrum()
    spec.table['mass'] = [200, 201, 202]
    spec.table['intensity'] = [1, 2, 3]
    spec_c = spec.copy()
    assert all(a==b for a,b in zip(spec.table['mass'], spec_c.table['mass']))
    spec_c = spec.normalize()
    assert all(a!=b for a,b in zip(spec.table['intensity'], spec_c.table['intensity']))

class Test_calc_mass:

    def test_calc_mass(self):
        spec = Spectrum.read_csv(sample1_path).drop_unassigned()
        calc_mass = spec.table['calc_mass'].round(6).to_list()
        spec.table = spec.table.drop(columns='calc_mass')
        spec = spec.calc_mass()
        assert all(a==b for a,b in zip(calc_mass, spec.table['calc_mass']))

    def test_calc_values(self):
        df = pd.DataFrame({'mass':[516.114752], 
                          'intensity':[1], 
                          'C':[24], 
                          'H':[23], 
                          'O':[12], 
                          'N':[1],
                          'S':[0], 
                          'assign':[True]})
        spec = Spectrum(table=df)
        spec = spec.calc_mass()
        assert spec.table.loc[0, 'calc_mass'] == 517.122029

class Test_calc_error:

    def test_calc_error(self):
        spec = Spectrum.read_csv(sample1_path).drop_unassigned()
        calc_mass = spec.table['rel_error'].round(6).to_list()
        spec.table = spec.table.drop(columns='rel_error')
        spec = spec.calc_error()
        calc_mass_n = spec.table['rel_error'].round(6).to_list()
        assert all(a==b for a,b in zip(calc_mass, calc_mass_n))

    @pytest.mark.parametrize('mass, result', [(516.114752, 0),
                                              (517.122029, 0),
                                              (517.121480, 0),
                                              (516.114944, 0.37)])
    def test_calc_values(self, mass, result):
        df = pd.DataFrame({'mass':[mass], 
                          'intensity':[1], 
                          'C':[24], 
                          'H':[23], 
                          'O':[12], 
                          'N':[1],
                          'S':[0], 
                          'assign':[True]})
        spec = Spectrum(table=df)
        spec = spec.calc_error()
        assert spec.table.loc[0, 'rel_error'].round(2) == result