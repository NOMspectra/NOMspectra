#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nomspectra. 
#
#    nomspectra is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomspectra is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomspectra.  If not, see <http://www.gnu.org/licenses/>.

from typing import Sequence, Optional
from functools import wraps
import numpy as np
import pandas as pd
import copy
from functools import lru_cache
from frozendict import frozendict

def _freeze(func):
    """
    freeze dict in func
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped

@_freeze
@lru_cache(maxsize=None)
def brutto_gen(elems: Optional[dict] = None, rules: bool = True) -> pd.DataFrame:
    """
    Generete brutto formula dataframe

    Parameters
    ----------
    elems: dict
        Dictonary with elements and their range for generate brutto table 
        Example: {'C':(1,60),'O_18':(0,3)} - content of carbon (main isotope) from 1 to 59,
        conent of isotope 18 oxygen from 0 to 2. 
        By default it is {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}
    rules: bool
        Rules: 0.25<H/C<2.2, O/C < 1, nitogen parity, DBE-O <= 10. 
        By default it is on, but for tmds should be off

    Returns
    -------
    pandas Dataframe
        Dataframe with masses for elements content
    """

    if elems is None:
        elems = {'C':(4, 51),'H':(4, 101),'O':(0,26), 'N':(0,4), 'S':(0,3)}

    #load elements table. Generatete in mass folder
    elems_mass_table = elements_table()
    elems_arr = []
    elems_dict = {}
    for el in elems:
        elems_arr.append(np.array(range(elems[el][0],elems[el][1])))
        if '_' not in el:
            temp = elems_mass_table.loc[elems_mass_table['element']==el].sort_values(by='abundance',ascending=False).reset_index(drop=True)
            elems_dict[el] = temp.loc[0,'mass']
        else:
            temp = elems_mass_table.loc[elems_mass_table['element_isotop']==el].reset_index(drop=True)
            elems_dict[el] = temp.loc[0,'mass']

    #generate grid with all possible combination of elements in their ranges
    t = np.array(np.meshgrid(*elems_arr)).T.reshape(-1,len(elems_arr))
    gdf = pd.DataFrame(t,columns=list(elems_dict.keys()))
    #do rules H/C, O/C, and parity
    if rules:

        temp = copy.deepcopy(gdf)
        temp=_merge_isotopes(temp)

        if 'C' not in temp or 'H' not in temp or 'O' not in temp:
            raise Exception('For applying rules in brutto must be CHO elements or their isotopes')
        
        temp['H/C'] = temp['H']/temp['C']
        temp['O/C'] = temp['O']/temp['C']
        gdf = gdf.loc[(temp['H/C'] < 2.2) & (temp['H/C'] > 0.25) & (temp['O/C'] < 1)]

        if 'N' not in temp:
            temp['N'] = 0
        
        temp['DBE-O'] = 1.0 + temp["C"] - 0.5 * temp["H"] + 0.5 * temp['N'] - temp['O']
        gdf = gdf.loc[temp['DBE-O'] <= 10]
        
        if 'N' in temp:
            temp['parity'] = (temp['H'] + temp['N'])%2
            gdf = gdf.loc[temp['parity']==0]

    #calculate mass
    masses = np.array(list(elems_dict.values()))
    gdf['mass'] = gdf.multiply(masses).sum(axis=1)
    gdf['mass'] = np.round(gdf['mass'], 6)

    gdf = gdf.sort_values("mass").reset_index(drop=True)

    return gdf

def _merge_isotopes(gdf: pd.DataFrame) -> pd.DataFrame:
    """
    All isotopes will be merged and title as main.

    Return
    ------
    pandas Dataframe    
    """

    for el in gdf:
        res = el.split('_')
        if len(res) == 2:
            if res[0] not in gdf:
                gdf[res[0]] = 0
            gdf[res[0]] = gdf[res[0]] + gdf[el]
            gdf = gdf.drop(columns=[el]) 

    return gdf

def get_elements_masses(elems: Sequence[str]) -> np.array :
    """
    Get elements masses from list

    Parameters
    ----------
    elems: Sequence[str]
        List of elements. Example: ['C', 'H', 'N', 'C_13', 'O']

    Return
    ------
    numpy array
    """
    
    elements = elements_table()    
    elems_masses = []

    for el in elems:
        if '_' not in el:
            temp = elements.loc[elements['element']==el].sort_values(by='abundance',ascending=False).reset_index(drop=True)
            elems_masses.append(temp.loc[0,'mass'])
        else:
            temp = elements.loc[elements['element_isotop']==el].reset_index(drop=True)
            elems_masses.append(temp.loc[0,'mass'])

    return np.array(elems_masses)

def gen_from_brutto(table: pd.DataFrame) -> pd.DataFrame:
    """
    Generate mass from brutto table

    Parameters
    ----------
    table: pandas Dataframe
        table with elemnt contnent

    Return
    ------
    pandas DataFrame
        Dataframe with elements and masses
    """
    masses = get_elements_masses(table.columns)

    table["calc_mass"] = table.multiply(masses).sum(axis=1)
    table["calc_mass"] = np.round(table["calc_mass"], 6)
    table.loc[table["calc_mass"] == 0, "calc_mass"] = np.NaN

    return table

def elements_table() -> pd.DataFrame: 
    """
    Table with exact mass of element and their isotop abundance

    Return
    ------
    Pandas DataFrame 
        Dataframe with exact mass of element and their isotop abundance
    """

    return pd.DataFrame(
    columns = ['element', 'mass', 'abundance', 'isotop', 'element_isotop'],
        data = [["Al", 26.981538, 100.0, 27, "Al_27"],
                ["Sb", 120.903818, 57.21, 121, "Sb_121"],
                ["Sb", 122.904216, 42.79, 123, "Sb_123"],
                ["Ar", 35.967546, 0.3365, 36, "Ar_36"],
                ["Ar", 37.962732, 0.0632, 38, "Ar_38"],
                ["Ar", 39.962383, 99.6003, 40, "Ar_40"],
                ["As", 74.921596, 100.0, 75, "As_75"],
                ["Ba", 129.90631, 0.106, 130, "Ba_130"],
                ["Ba", 131.905056, 0.101, 132, "Ba_132"],
                ["Ba", 133.904503, 2.417, 134, "Ba_134"],
                ["Ba", 134.905683, 6.592, 135, "Ba_135"],
                ["Ba", 135.90457, 7.854, 136, "Ba_136"],
                ["Ba", 136.905821, 11.232, 137, "Ba_137"],
                ["Ba", 137.905241, 71.698, 138, "Ba_138"],
                ["Be", 9.012182, 100.0, 9, "Be_9"],
                ["Bi", 208.980383, 100.0, 209, "Bi_209"],
                ["B", 10.012937, 19.9, 10, "B_10"],
                ["B", 11.009305, 80.1, 11, "B_11"],
                ["Br", 78.918338, 50.69, 79, "Br_79"],
                ["Br", 80.916291, 49.31, 81, "Br_81"],
                ["Cd", 105.906458, 1.25, 106, "Cd_106"],
                ["Cd", 107.904183, 0.89, 108, "Cd_108"],
                ["Cd", 109.903006, 12.49, 110, "Cd_110"],
                ["Cd", 110.904182, 12.8, 111, "Cd_111"],
                ["Cd", 111.902757, 24.13, 112, "Cd_112"],
                ["Cd", 112.904401, 12.22, 113, "Cd_113"],
                ["Cd", 113.903358, 28.73, 114, "Cd_114"],
                ["Cd", 115.904755, 7.49, 116, "Cd_116"],
                ["Ca", 39.962591, 96.941, 40, "Ca_40"],
                ["Ca", 41.958618, 0.647, 42, "Ca_42"],
                ["Ca", 42.958767, 0.135, 43, "Ca_43"],
                ["Ca", 43.955481, 2.086, 44, "Ca_44"],
                ["Ca", 45.953693, 0.004, 46, "Ca_46"],
                ["Ca", 47.952534, 0.187, 48, "Ca_48"],
                ["C", 12.0, 98.93, 12, "C_12"],
                ["C", 13.003355, 1.07, 13, "C_13"],
                ["Ce", 135.907144, 0.185, 136, "Ce_136"],
                ["Ce", 137.905986, 0.251, 138, "Ce_138"],
                ["Ce", 139.905434, 88.45, 140, "Ce_140"],
                ["Ce", 141.90924, 11.114, 142, "Ce_142"],
                ["Cs", 132.905447, 100.0, 133, "Cs_133"],
                ["Cl", 34.968853, 75.78, 35, "Cl_35"],
                ["Cl", 36.965903, 24.22, 37, "Cl_37"],
                ["Cr", 49.94605, 4.345, 50, "Cr_50"],
                ["Cr", 51.940512, 83.789, 52, "Cr_52"],
                ["Cr", 52.940654, 9.501, 53, "Cr_53"],
                ["Cr", 53.938885, 2.365, 54, "Cr_54"],
                ["Co", 58.9332, 100.0, 59, "Co_59"],
                ["Cu", 62.929601, 69.17, 63, "Cu_63"],
                ["Cu", 64.927794, 30.83, 65, "Cu_65"],
                ["Dy", 155.924278, 0.06, 156, "Dy_156"],
                ["Dy", 157.924405, 0.1, 158, "Dy_158"],
                ["Dy", 159.925194, 2.34, 160, "Dy_160"],
                ["Dy", 160.92693, 18.91, 161, "Dy_161"],
                ["Dy", 161.926795, 25.51, 162, "Dy_162"],
                ["Dy", 162.928728, 24.9, 163, "Dy_163"],
                ["Dy", 163.929171, 28.18, 164, "Dy_164"],
                ["Er", 161.928775, 0.14, 162, "Er_162"],
                ["Er", 163.929197, 1.61, 164, "Er_164"],
                ["Er", 165.93029, 33.61, 166, "Er_166"],
                ["Er", 166.932045, 22.93, 167, "Er_167"],
                ["Er", 167.932368, 26.78, 168, "Er_168"],
                ["Er", 169.93546, 14.93, 170, "Er_170"],
                ["Eu", 150.919846, 47.81, 151, "Eu_151"],
                ["Eu", 152.921226, 52.19, 153, "Eu_153"],
                ["F", 18.998403, 100.0, 19, "F_19"],
                ["Ga", 68.925581, 60.108, 69, "Ga_69"],
                ["Ga", 70.924705, 39.892, 71, "Ga_71"],
                ["Gd", 151.919788, 0.2, 152, "Gd_152"],
                ["Gd", 153.920862, 2.18, 154, "Gd_154"],
                ["Gd", 154.822619, 14.8, 155, "Gd_155"],
                ["Gd", 155.92212, 20.47, 156, "Gd_156"],
                ["Gd", 156.923957, 15.65, 157, "Gd_157"],
                ["Gd", 157.924101, 24.84, 158, "Gd_158"],
                ["Gd", 159.927051, 21.86, 160, "Gd_160"],
                ["Ge", 69.92425, 20.84, 70, "Ge_70"],
                ["Ge", 71.922076, 27.54, 72, "Ge_72"],
                ["Ge", 72.923459, 7.73, 73, "Ge_73"],
                ["Ge", 73.921178, 36.5, 74, "Ge_74"],
                ["Ge", 75.921403, 7.61, 76, "Ge_76"],
                ["Au", 196.966552, 100.0, 197, "Au_197"],
                ["Hf", 173.94004, 0.16, 174, "Hf_174"],
                ["Hf", 175.941402, 5.26, 176, "Hf_176"],
                ["Hf", 176.94322, 18.6, 177, "Hf_177"],
                ["Hf", 177.943698, 27.28, 178, "Hf_178"],
                ["Hf", 178.945815, 13.62, 179, "Hf_179"],
                ["Hf", 179.946549, 35.08, 180, "Hf_180"],
                ["He", 3.016029, 0.000137, 3, "He_3"],
                ["He", 4.002603, 99.999863, 4, "He_4"],
                ["Ho", 164.930319, 100.0, 165, "Ho_165"],
                ["H", 1.007825, 99.9885, 1, "H_1"],
                ["H", 2.014102, 0.115, 2, "H_2"],
                ["In", 112.904061, 4.29, 113, "In_113"],
                ["In", 114.903878, 95.71, 115, "In_115"],
                ["I", 126.904468, 100.0, 127, "I_127"],
                ["Ir", 190.960591, 37.3, 191, "Ir_191"],
                ["Ir", 192.962924, 62.7, 193, "Ir_193"],
                ["Fe", 53.939615, 5.845, 54, "Fe_54"],
                ["Fe", 55.934942, 91.754, 56, "Fe_56"],
                ["Fe", 56.935399, 2.119, 57, "Fe_57"],
                ["Fe", 57.93328, 0.282, 58, "Fe_58"],
                ["Kr", 77.920386, 0.35, 78, "Kr_78"],
                ["Kr", 79.916378, 2.28, 80, "Kr_80"],
                ["Kr", 81.913485, 11.58, 82, "Kr_82"],
                ["Kr", 82.914136, 11.49, 83, "Kr_83"],
                ["Kr", 83.911507, 57.0, 84, "Kr_84"],
                ["Kr", 85.91061, 17.3, 86, "Kr_86"],
                ["La", 137.907107, 0.09, 138, "La_138"],
                ["La", 138.906348, 99.91, 139, "La_139"],
                ["Pb", 203.973029, 1.4, 204, "Pb_204"],
                ["Pb", 205.974449, 24.1, 206, "Pb_206"],
                ["Pb", 206.975881, 22.1, 207, "Pb_207"],
                ["Pb", 207.976636, 52.4, 208, "Pb_208"],
                ["Li", 6.015122, 7.59, 6, "Li_6"],
                ["Li", 7.016004, 92.41, 7, "Li_7"],
                ["Lu", 174.940768, 97.41, 175, "Lu_175"],
                ["Lu", 175.942682, 2.59, 176, "Lu_176"],
                ["Mg", 23.985042, 78.99, 24, "Mg_24"],
                ["Mg", 24.985837, 10.0, 25, "Mg_25"],
                ["Mg", 25.982593, 11.01, 26, "Mg_26"],
                ["Mn", 54.93805, 100.0, 55, "Mn_55"],
                ["Hg", 195.965815, 0.15, 196, "Hg_196"],
                ["Hg", 197.966752, 9.97, 198, "Hg_198"],
                ["Hg", 198.968262, 16.87, 199, "Hg_199"],
                ["Hg", 199.968309, 23.1, 200, "Hg_200"],
                ["Hg", 200.970285, 13.18, 201, "Hg_201"],
                ["Hg", 201.970626, 29.86, 202, "Hg_202"],
                ["Hg", 203.973476, 6.87, 204, "Hg_204"],
                ["Mo", 91.90681, 14.84, 92, "Mo_92"],
                ["Mo", 93.905088, 9.25, 94, "Mo_94"],
                ["Mo", 94.905841, 15.92, 95, "Mo_95"],
                ["Mo", 95.904679, 16.68, 96, "Mo_96"],
                ["Mo", 96.906021, 9.55, 97, "Mo_97"],
                ["Mo", 97.905408, 24.13, 98, "Mo_98"],
                ["Mo", 99.907477, 9.63, 100, "Mo_100"],
                ["Nd", 141.907719, 27.2, 142, "Nd_142"],
                ["Nd", 142.90981, 12.2, 143, "Nd_143"],
                ["Nd", 143.910083, 23.8, 144, "Nd_144"],
                ["Nd", 144.912569, 8.3, 145, "Nd_145"],
                ["Nd", 145.913112, 17.2, 146, "Nd_146"],
                ["Nd", 147.916889, 5.7, 148, "Nd_148"],
                ["Nd", 149.920887, 5.6, 150, "Nd_150"],
                ["Ne", 19.99244, 90.48, 20, "Ne_20"],
                ["Ne", 20.993847, 0.27, 21, "Ne_21"],
                ["Ne", 21.991386, 9.25, 22, "Ne_22"],
                ["Ni", 57.935348, 68.0769, 58, "Ni_58"],
                ["Ni", 59.930791, 26.2231, 60, "Ni_60"],
                ["Ni", 60.93106, 1.1399, 61, "Ni_61"],
                ["Ni", 61.928349, 3.6345, 62, "Ni_62"],
                ["Ni", 63.92797, 0.9256, 64, "Ni_64"],
                ["Nb", 92.906378, 100.0, 93, "Nb_93"],
                ["N", 14.003074, 99.632, 14, "N_14"],
                ["N", 15.000109, 0.368, 15, "N_15"],
                ["Os", 183.952491, 0.02, 184, "Os_184"],
                ["Os", 185.953838, 1.59, 186, "Os_186"],
                ["Os", 186.955748, 1.96, 187, "Os_187"],
                ["Os", 187.955836, 13.24, 188, "Os_188"],
                ["Os", 188.958145, 16.15, 189, "Os_189"],
                ["Os", 189.958445, 26.26, 190, "Os_190"],
                ["Os", 191.961479, 40.78, 192, "Os_192"],
                ["O", 15.994915, 99.757, 16, "O_16"],
                ["O", 16.999132, 0.038, 17, "O_17"],
                ["O", 17.99916, 0.205, 18, "O_18"],
                ["Pd", 101.905608, 1.02, 102, "Pd_102"],
                ["Pd", 103.904035, 11.14, 104, "Pd_104"],
                ["Pd", 104.905084, 22.33, 105, "Pd_105"],
                ["Pd", 105.903483, 27.33, 106, "Pd_106"],
                ["Pd", 107.903894, 26.46, 108, "Pd_108"],
                ["Pd", 109.905152, 11.72, 110, "Pd_110"],
                ["P", 30.973762, 100.0, 31, "P_31"],
                ["Pt", 189.95993, 0.014, 190, "Pt_190"],
                ["Pt", 191.961035, 0.782, 192, "Pt_192"],
                ["Pt", 193.962664, 32.967, 194, "Pt_194"],
                ["Pt", 194.964774, 33.832, 195, "Pt_195"],
                ["Pt", 195.964935, 25.242, 196, "Pt_196"],
                ["Pt", 197.967876, 7.163, 198, "Pt_198"],
                ["K", 38.963707, 93.2581, 39, "K_39"],
                ["K", 39.963999, 0.0117, 40, "K_40"],
                ["K", 40.961826, 6.7302, 41, "K_41"],
                ["Pr", 140.907648, 100.0, 141, "Pr_141"],
                ["Re", 184.952956, 37.4, 185, "Re_185"],
                ["Re", 186.955751, 62.6, 187, "Re_187"],
                ["Rh", 102.905504, 100.0, 103, "Rh_103"],
                ["Rb", 84.911789, 72.17, 85, "Rb_85"],
                ["Rb", 86.909183, 27.83, 87, "Rb_87"],
                ["Ru", 95.907598, 5.54, 96, "Ru_96"],
                ["Ru", 97.905287, 1.87, 98, "Ru_98"],
                ["Ru", 98.905939, 12.76, 99, "Ru_99"],
                ["Ru", 99.90422, 12.6, 100, "Ru_100"],
                ["Ru", 100.905582, 17.06, 101, "Ru_101"],
                ["Ru", 101.90435, 31.55, 102, "Ru_102"],
                ["Ru", 103.90543, 18.62, 104, "Ru_104"],
                ["Sm", 143.911995, 3.07, 144, "Sm_144"],
                ["Sm", 146.914893, 14.99, 147, "Sm_147"],
                ["Sm", 147.914818, 11.24, 148, "Sm_148"],
                ["Sm", 148.91718, 13.82, 149, "Sm_149"],
                ["Sm", 149.917271, 7.38, 150, "Sm_150"],
                ["Sm", 151.919728, 26.75, 152, "Sm_152"],
                ["Sm", 153.922205, 22.75, 154, "Sm_154"],
                ["Sc", 44.95591, 100.0, 45, "Sc_45"],
                ["Se", 73.922477, 0.89, 74, "Se_74"],
                ["Se", 75.919214, 9.37, 76, "Se_76"],
                ["Se", 76.919915, 7.63, 77, "Se_77"],
                ["Se", 77.91731, 23.77, 78, "Se_78"],
                ["Se", 79.916522, 49.61, 80, "Se_80"],
                ["Se", 81.9167, 8.73, 82, "Se_82"],
                ["Si", 27.976927, 92.2297, 28, "Si_28"],
                ["Si", 28.976495, 4.6832, 29, "Si_29"],
                ["Si", 29.97377, 3.0872, 30, "Si_30"],
                ["Ag", 106.905093, 51.839, 107, "Ag_107"],
                ["Ag", 108.904756, 48.161, 109, "Ag_109"],
                ["Na", 22.98977, 100.0, 23, "Na_23"],
                ["Sr", 83.913425, 0.56, 84, "Sr_84"],
                ["Sr", 85.909262, 9.86, 86, "Sr_86"],
                ["Sr", 86.908879, 7.0, 87, "Sr_87"],
                ["Sr", 87.905614, 82.58, 88, "Sr_88"],
                ["S", 31.972071, 94.93, 32, "S_32"],
                ["S", 32.971458, 0.76, 33, "S_33"],
                ["S", 33.967867, 4.29, 34, "S_34"],
                ["S", 35.967081, 0.02, 36, "S_36"],
                ["Ta", 179.947466, 0.012, 180, "Ta_180"],
                ["Ta", 180.947996, 99.988, 181, "Ta_181"],
                ["Te", 119.90402, 0.09, 120, "Te_120"],
                ["Te", 121.903047, 2.55, 122, "Te_122"],
                ["Te", 122.904273, 0.89, 123, "Te_123"],
                ["Te", 123.902819, 4.74, 124, "Te_124"],
                ["Te", 124.904425, 7.07, 125, "Te_125"],
                ["Te", 125.903306, 18.84, 126, "Te_126"],
                ["Te", 127.904461, 31.74, 128, "Te_128"],
                ["Te", 129.906223, 34.08, 130, "Te_130"],
                ["Tb", 158.925343, 100.0, 159, "Tb_159"],
                ["Tl", 202.972329, 29.524, 203, "Tl_203"],
                ["Tl", 204.974412, 70.476, 205, "Tl_205"],
                ["Th", 232.03805, 100.0, 232, "Th_232"],
                ["Tm", 168.934211, 100.0, 169, "Tm_169"],
                ["Sn", 111.904821, 0.97, 112, "Sn_112"],
                ["Sn", 113.902782, 0.66, 114, "Sn_114"],
                ["Sn", 114.903346, 0.34, 115, "Sn_115"],
                ["Sn", 115.901744, 14.54, 116, "Sn_116"],
                ["Sn", 116.902954, 7.68, 117, "Sn_117"],
                ["Sn", 117.901606, 24.22, 118, "Sn_118"],
                ["Sn", 118.903309, 8.59, 119, "Sn_119"],
                ["Sn", 119.902197, 32.58, 120, "Sn_120"],
                ["Sn", 121.90344, 4.63, 122, "Sn_122"],
                ["Sn", 123.905275, 5.79, 124, "Sn_124"],
                ["Ti", 45.952629, 8.25, 46, "Ti_46"],
                ["Ti", 46.951764, 7.44, 47, "Ti_47"],
                ["Ti", 47.947871, 73.72, 48, "Ti_48"],
                ["W", 179.946706, 0.12, 180, "W_180"],
                ["W", 181.948206, 26.5, 182, "W_182"],
                ["W", 182.950224, 14.31, 183, "W_183"],
                ["W", 183.950933, 30.64, 184, "W_184"],
                ["W", 185.954362, 28.43, 186, "W_186"],
                ["U", 234.040946, 0.0055, 234, "U_234"],
                ["U", 235.043923, 0.72, 235, "U_235"],
                ["U", 238.050783, 99.2745, 238, "U_238"],
                ["V", 49.947163, 0.25, 50, "V_50"],
                ["V", 50.943964, 99.75, 51, "V_51"],
                ["Xe", 123.905896, 0.09, 124, "Xe_124"],
                ["Xe", 125.904269, 0.09, 126, "Xe_126"],
                ["Xe", 127.90353, 1.92, 128, "Xe_128"],
                ["Xe", 128.904779, 26.44, 129, "Xe_129"],
                ["Xe", 129.903508, 10.44, 130, "Xe_130"],
                ["Xe", 130.905082, 21.18, 131, "Xe_131"],
                ["Xe", 131.904154, 26.89, 132, "Xe_132"],
                ["Xe", 133.905395, 10.44, 134, "Xe_134"],
                ["Xe", 135.90722, 8.87, 136, "Xe_136"],
                ["Yb", 167.933894, 0.13, 168, "Yb_168"],
                ["Yb", 169.934759, 3.04, 170, "Yb_170"],
                ["Yb", 170.936322, 14.28, 171, "Yb_171"],
                ["Yb", 171.936378, 21.83, 172, "Yb_172"],
                ["Yb", 172.938207, 16.13, 173, "Yb_173"],
                ["Yb", 173.938858, 31.83, 174, "Yb_174"],
                ["Yb", 175.942568, 12.76, 176, "Yb_176"],
                ["Y", 88.905848, 100.0, 89, "Y_89"],
                ["Zn", 63.929147, 48.63, 64, "Zn_64"],
                ["Zn", 65.926037, 27.9, 66, "Zn_66"],
                ["Zn", 66.927131, 4.1, 67, "Zn_67"],
                ["Zn", 67.924848, 18.75, 68, "Zn_68"],
                ["Zn", 69.925325, 0.62, 70, "Zn_70"],
                ["Zr", 89.904704, 51.45, 90, "Zr_90"],
                ["Zr", 90.905645, 11.22, 91, "Zr_91"],
                ["Zr", 91.90504, 17.15, 92, "Zr_92"],
                ["Zr", 93.906316, 17.38, 94, "Zr_94"],
                ["Zr", 95.908276, 2.8, 96, "Zr_96"]])

if __name__ == '__main__':
    print(brutto_gen())