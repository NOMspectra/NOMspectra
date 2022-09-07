from nhsmass.spectrum import Spectrum
import pytest
import os
import numpy as np
import pandas as pd

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'test', 'sample1.txt')
sample2_path = os.path.join(root, 'test', 'sample2.txt')

spec1 = Spectrum.read_csv(sample1_path, take_only_mz=True)
spec1 = spec1.assign().drop_unassigned()
mapper = {'m/z':'mass','I':'intensity'}
spec2 = Spectrum.read_csv(sample2_path, mapper=mapper, take_only_mz=True, sep='\t')
spec2 = spec2.assign().drop_unassigned()

def test_or():
    assert len(spec1 + spec1) == len(spec1)
    assert len(spec1.__or__(spec1)) == len(spec1)
    assert len(spec1 + spec2) == 6634

def test_xor():
    assert len(spec1 ^ spec1) == 0
    assert len(spec1.__xor__(spec1)) == 0
    assert len(spec1 ^ spec2) == 3141

def test_and():
    assert len(spec1 & spec1) == len(spec1)
    assert len(spec1.__and__(spec1)) == len(spec1)
    assert len(spec1 & spec2) == 3493

def test_sub():
    assert len(spec1 - spec1) == 0
    assert len(spec1.__sub__(spec1)) == 0
    assert len(spec1 - spec2) == 530

def test_intens_sub():
    assert len(spec1.intens_sub(spec1)) == 0
    assert len(spec1.intens_sub(spec2)) == 1692

@pytest.mark.parametrize('metric, result', [("tanimoto", 0.862),
                                            ("jaccard", 0.527),
                                            ("cosine", 0.93)])
def test_simmilarity(metric, result):
    assert round(spec1.simmilarity(spec1, mode=metric),3) == 1
    assert round(spec1.simmilarity(spec2, mode=metric),3) == result