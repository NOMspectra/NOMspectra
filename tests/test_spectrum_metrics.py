from nomhsms.spectrum import Spectrum
import pytest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

root = os.path.join(os.path.dirname(__file__), '..')
sample1_path = os.path.join(root, 'tests', 'sample1.txt')
spec = Spectrum.read_csv(sample1_path, assign_mark=True).drop_unassigned()

@pytest.mark.parametrize('el, result', [('S','C24H23O12NS3'),('C_13','C24H23O12N(C_13)3')])
def test_brutto(el, result):
    df = pd.DataFrame({'mass':[516.114752], 
                        'intensity':[1], 
                        'C':[24], 
                        'H':[23], 
                        'O':[12], 
                        'N':[1],
                        el:[3], 
                        'assign':[True]})
    spec = Spectrum(table=df)
    spec = spec.brutto()
    assert spec.table.loc[0, 'brutto'] == result

@pytest.mark.parametrize('el, result', [((6, 6, 5, 0), True),
                                        ((6, 6, 0, 0), False)])
def test_cram(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.cram()
    assert spec.table.loc[0, 'CRAM'] == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 0.5),
                                        ((6, 6, 5, 0), -1)])
def test_ai(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.ai()
    assert spec.table.loc[0, 'AI'] == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 4),
                                        ((6, 6, 5, 0), 1)])
def test_cai(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.cai()
    assert spec.table.loc[0, 'CAI'] == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 6),
                                        ((6, 6, 5, 0), 4)])
def test_dbe(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.dbe()
    assert spec.table.loc[0, 'DBE'] == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 2),
                                        ((6, 6, 5, 0), -1)])
def test_dbe_o(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.dbe_o()
    assert spec.table.loc[0, 'DBE-O'] == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 0.25),
                                        ((6, 6, 5, 0), -0.167)])
def test_dbe_oc(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.dbe_oc()
    assert round(spec.table.loc[0, 'DBE-OC'],3) == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), (0.75, 0.5)),
                                        ((6, 6, 5, 0), (1, 0.833))])
def test_hc_oc(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.hc_oc()
    assert round(spec.table.loc[0, 'H/C'],3) == result[0]
    assert round(spec.table.loc[0, 'O/C'],3) == result[1]

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), (165.841223, 0.158777)),
                                        ((6, 6, 5, 0), (157.845077, 0.154923))])
def test_kendrick(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.kendrick()
    assert round(spec.table.loc[0, 'Ke'],6) == result[0]
    assert round(spec.table.loc[0, 'KMD'],6) == result[1]

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 0.25),
                                        ((6, 6, 5, 0), 0.667)])
def test_nosc(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.nosc()
    assert round(spec.table.loc[0, 'NOSC'],3) == result

@pytest.mark.parametrize('el, result', [((8, 6, 4, 0), 'undefinded'),
                                        ((6, 6, 5, 0), 'unsat_highOC'),
                                        ((12, 18, 3, 0), 'lipids'),
                                        ((8, 15, 1, 3), 'N-satureted'),
                                        ((7, 12, 5, 0), 'aliphatics'),
                                        ((8, 8, 4, 0), 'unsat_lowOC'),
                                        ((11, 8, 4, 0), 'aromatic_lowOC'),
                                        ((9, 4, 6, 0), 'aromatic_highOC'),
                                        ((10, 6, 3, 0), 'condensed_lowOC'),
                                        ((9, 4, 5, 0), 'condensed_highOC')])
def test_mol_class(el, result):
    df = pd.DataFrame({'mass':[1], 'intensity':[1], 'assign':[True],
                        'C':[el[0]], 
                        'H':[el[1]], 
                        'O':[el[2]], 
                        'N':[el[3]]})
    spec = Spectrum(table=df)
    spec = spec.mol_class()
    assert spec.table.loc[0, 'class'] == result

def test_get_mol_class_density():
    density = [0.469540, 0.458127, 0.000994, 0.004782, 0.008596, 0.002360, 0.025681, 0.004432, 0.015814, 0.009675]
    values = spec.get_mol_class()['density'].to_list()
    assert all(round(a,3)==round(b,3) for a,b in zip(density, values))

def test_dbe_vs_o():
    values = spec.get_dbe_vs_o(olim=(5,15))
    assert round(values[0],3 ) == 0.736

def test_get_squares_vk():
    density = [0.0,
                0.0007,
                0.0034,
                0.0024,
                0.0019,
                0.0042,
                0.0615,
                0.2934,
                0.0611,
                0.0012,
                0.013,
                0.1883,
                0.2816,
                0.027,
                0.0018,
                0.0019,
                0.018,
                0.0253,
                0.009,
                0.0043]
    values = spec.get_squares_vk()['value'].to_list()
    assert all(round(a,4)==round(b,4) for a,b in zip(density, values))

def test_get_mol_metrics():
    metrics = ['AI','C','CAI','CRAM','C_13','DBE',
                'DBE-O','DBE-OC','DBE_AI','H','H/C',
                'N','NOSC','O','O/C','mass']
    values = [-0.205, 18.866, 9.212, 0.692, 0.323, 9.555,
              -0.257, -0.019, -0.422, 21.431, 1.118, 0.164,
              -0.052, 9.813, 0.52, 410.43]
    data = spec.get_mol_metrics()
    assert all(a==b for a,b in zip(metrics, data['metric'].to_list()))
    assert all(round(a,3)==round(b,3) for a,b in zip(values, data['value'].to_list()))

@pytest.mark.parametrize('how, result', [('weight',9.555), 
                                         ('mean',10.802), 
                                         ('median',10.0), 
                                         ('max',29.0), 
                                         ('min',-3.0), 
                                         ('std',4.387)])
def test_get_mol_metrics_how(how, result):
    value = spec.get_mol_metrics(func=how, metrics=['DBE']).loc[0,'value']
    assert round(value,3) == result