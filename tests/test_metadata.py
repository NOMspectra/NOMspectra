from natorgms.metadata import MetaData
from natorgms.spectrum import Spectrum
import pytest
import os

@pytest.mark.parametrize('d, result', [
   (None, {}),
   ({'name':'test', 'sign':'-'}, {'name':'test', 'sign':'-'}),
   ({'NaMe':'test', 'sigN':'-','val':10}, {'name':'test', 'sign':'-','val':10})])
def test_metadata_init(d, result):
    md = MetaData(d)
    assert md == result

@pytest.mark.parametrize('d, add, result', [
    (None, {'name':'test'} , {'name':'test'}),
    ({'name':'test'} , {'name':'test2'} ,{'name':'test2'}),
   ({'name':'test', 'sign':'-'}, {'val':10}, {'name':'test', 'sign':'-', 'val':10})])
def test_metadata_add(d, add, result):
    md = MetaData(d)
    md.add(add)
    assert md == result

def test_metadata_combine():
    spec1 = Spectrum(metadata={'name':'test1'})
    spec2 = Spectrum(metadata={'name':'test2'})
    assert MetaData.combine_two_name(spec1, spec2) == 'test1_test2'