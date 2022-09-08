from nhsmass.spectrum import Spectrum
from nhsmass.chem import Reaction
import pytest
import os

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
sample2_path = os.path.join(root, 'tests', 'sample2.txt')
spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True).assign().drop_unassigned()
mapper = {'m/z':'mass','I':'intensity'}
spec2 = Spectrum.read_csv(sample2_path, mapper=mapper, take_only_mz=True, sep='\t').assign().drop_unassigned()

def test_reaction_init():
    r = Reaction(spec1, spec2)
    assert r.sourse.metadata['name'] == 'sample1'
    assert r.product.metadata['name'] == 'sample2'

def test_find_modification():
    r = Reaction(spec1, spec2)
    res = r.find_modification({'C':(4,5),'H':(4,5),'O':(2,3)})
    sm = res.sourse.table.loc[res.sourse.table['modified'] == True].reset_index(drop=True)
    pm = res.product.table.loc[res.product.table['modified'] == True].reset_index(drop=True)
    res.draw_modification()
    assert len(sm) == len(pm)
    assert round(pm['mass'].mean() - sm['mass'].mean(), 3) == 84.021