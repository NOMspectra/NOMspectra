from nomspectra.spectrum import Spectrum
from nomspectra.spectra import SpectrumList
import pytest
import os

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
sample2_path = os.path.join(root, 'tests', 'sample2.txt')
spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True).assign().drop_unassigned()
mapper = {'m/z':'mass','I':'intensity'}
spec2 = Spectrum.read_csv(sample2_path, mapper=mapper, take_only_mz=True, sep='\t').assign().drop_unassigned()
folder = os.path.join(root, 'tests', 'temp')
if 'temp' not in os.listdir(os.path.join(root, 'tests')):
    os.mkdir(folder)

class Test_load_and_save:

    def test_init(self):
        specs = SpectrumList([spec1, spec2])
        assert len(specs) == 2

    def test_names(self):
        specs = SpectrumList([spec1, spec2])
        names = specs.get_names()
        assert names[0] == 'sample1'
        assert names[1] == 'sample2'

    def test_to_csv(self):
        specs = SpectrumList([spec1, spec2])
        specs.to_csv(folder)
        res = os.listdir(folder)
        assert 'sample1.csv' in res
        assert 'sample2.csv' in res

    def test_to_json(self):
        specs = SpectrumList([spec1, spec2])
        os.path.join(folder, 'test.json')
        specs.to_json(os.path.join(folder, 'test.json'))
        res = os.listdir(folder)
        assert 'test.json' in res

    def test_read_csv(self):
        specs = SpectrumList.read_csv(folder)
        assert len(specs) == 2

    def test_read_json(self):
        specs = SpectrumList.read_json(os.path.join(folder, 'test.json'))
        assert len(specs) == 2

@pytest.mark.parametrize('metric, result', [("tanimoto", 0.862),
                                            ("jaccard", 0.527),
                                            ("cosine", 0.93)])
def test_get_simmilarity(metric, result):
    specs = SpectrumList([spec1, spec2])
    res = specs.get_simmilarity(mode=metric)
    assert len(res) == 2
    assert res[0,0] == 1
    assert res[1,1] == 1
    assert res[0,1] == res[1,0]
    assert round(res[0,1], 3) == result

def test_get_mol_metrics():
    specs = SpectrumList([spec1, spec2])
    res = specs.get_mol_metrics()
    index = set(['AI', 'C', 'CAI', 'CRAM', 'DBE', 'DBE-O', 'DBE-OC', 'DBE_AI', 'H',
                'H/C', 'N', 'NOSC', 'O', 'O/C', 'S', 'mass'])
    assert index == set(res.index.to_list())
    assert set(['sample1', 'sample2']) == set(res.columns.to_list())

def test_get_mol_metrics_dbe():
    specs = SpectrumList([spec1, spec2])
    res = specs.get_mol_metrics(metrics=['DBE'])
    assert round(res.loc['DBE','sample1'], 3) == 9.828
    assert round(res.loc['DBE','sample2'], 3) == 10.393

def test_get_square_vk():
    specs = SpectrumList([spec1, spec2])
    res = specs.get_square_vk()
    assert set(['sample1', 'sample2']) == set(res.columns.to_list())
    assert round(res.loc[7,'sample1'], 3) == 0.083
    assert round(res.loc[7,'sample2'], 3) == 0.089

def test_get_mol_density():
    specs = SpectrumList([spec1, spec2])
    res = specs.get_mol_density()
    index = set(['unsat_lowOC', 'unsat_highOC', 'condensed_lowOC', 'condensed_highOC',
                'aromatic_lowOC', 'aromatic_highOC', 'aliphatics', 'lipids',
                'N-satureted', 'undefinded'])
    assert index == set(res.index.to_list())
    assert set(['sample1', 'sample2']) == set(res.columns.to_list())
    assert round(res.loc['unsat_highOC','sample1'], 3) == 0.42
    assert round(res.loc['unsat_highOC','sample2'], 3) == 0.385