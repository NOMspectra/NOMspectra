import nhsmass.brutto as brutto
import pytest
import pandas as pd


@pytest.mark.parametrize('elems, rules, result', [
    (None, True, 244362),
    ({'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}, True, 244362),
    ({'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}, False, 1422408),
    ({'C':(1,3),'H':(1,3),'O':(1,3)}, True, 1),
    ({'C':(1,4),'H':(1,3),'O':(1,3)}, True, 3),
    ({'C':(1,3),'H':(1,3),'O':(1,3), 'C_13':(0,2)}, True, 4),
    ({'C':(1,3),'H':(1,3),'O':(1,3)}, False, 8)])
def test_brutto_gen_len(elems, rules, result):
    res = brutto.brutto_gen(elems, rules)
    assert len(res) == result
    if elems is not None:
        for i in elems.keys():
            assert i in res.columns

@pytest.mark.parametrize('elems, rules, result', [
    (None, True, 52.0313),
    ({'C':(1,3),'H':(1,3),'O':(1,3)}, True, 42.010565),
    ({'C_13':(1,3),'H':(1,3),'O':(1,3)}, False, 30.006095),
    ({'C':(1,3),'H':(1,3),'O':(1,3)}, False, 29.00274),
    ({'C':(1,3),'H_2':(1,3),'O':(1,3)}, False, 30.009017)])
def test_brutto_gen_mass(elems, rules, result):
    res = brutto.brutto_gen(elems, rules)
    assert res.loc[0,'mass'] == result


def test_brutto_merge_isotopes():
    df = pd.DataFrame({'C':[1], 'C_13':[2], 'H':[2], 'H_2':[4], 'O':[3]})
    res = brutto._merge_isotopes(df)
    assert set(['C', 'H', 'O']) == set(res.columns)
    assert res.loc[0,'C'] == 3
    assert res.loc[0,'H'] == 6
    assert res.loc[0,'O'] == 3

@pytest.mark.parametrize('elems, result', [
    (['C'], [12]),
    (['C_13'], [13.003355]),
    (['C','C_13'], [12, 13.003355]),
    (['C','H','O','N','S','P'], [12, 1.007825, 15.994915, 14.003074, 31.972071, 30.973762])])
def test_brutto_get_element_mass(elems, result):
    res = brutto.get_elements_masses(elems)
    assert all(a==b for a,b in zip(result, res))

def test_brutto_gen_from_brutto():
    df = pd.DataFrame({'C':[1, 1], 'C_13':[1, 0], 'H':[2, 2], 'O':[3, 3]})
    brutto.gen_from_brutto(df)
    assert df.loc[0,'calc_mass'] == 75.00375
    assert df.loc[1,'calc_mass'] == 62.000395

def test_brutto_elems_table():
    df = brutto.elements_table()
    assert len(df.loc[df['element'] == 'C']) == 2
    assert len(df) == 285
    assert set(df.columns) == set(['element','mass','abundance','isotop','element_isotop'])