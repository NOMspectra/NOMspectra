#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nomhsms. 
#
#    nomhsms is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomhsms is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomhsms.  If not, see <http://www.gnu.org/licenses/>.

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
    data = [["Al",26.981538,100.0,26,"Al_26"],
            ["Sb",120.903818,57.21,120,"Sb_120"],
            ["Sb",122.904216,42.79,122,"Sb_122"],
            ["Ar",35.967546,0.3365,35,"Ar_35"],
            ["Ar",37.962732,0.0632,37,"Ar_37"],
            ["Ar",39.962383,99.6003,39,"Ar_39"],
            ["As",74.921596,100.0,74,"As_74"],
            ["Ba",129.90631,0.106,129,"Ba_129"],
            ["Ba",131.905056,0.101,131,"Ba_131"],
            ["Ba",133.904503,2.417,133,"Ba_133"],
            ["Ba",134.905683,6.592,134,"Ba_134"],
            ["Ba",135.90457,7.854,135,"Ba_135"],
            ["Ba",136.905821,11.232,136,"Ba_136"],
            ["Ba",137.905241,71.698,137,"Ba_137"],
            ["Be",9.012182,100.0,9,"Be_9"],
            ["Bi",208.980383,100.0,208,"Bi_208"],
            ["B",10.012937,19.9,10,"B_10"],
            ["B",11.009305,80.1,11,"B_11"],
            ["Br",78.918338,50.69,78,"Br_78"],
            ["Br",80.916291,49.31,80,"Br_80"],
            ["Cd",105.906458,1.25,105,"Cd_105"],
            ["Cd",107.904183,0.89,107,"Cd_107"],
            ["Cd",109.903006,12.49,109,"Cd_109"],
            ["Cd",110.904182,12.8,110,"Cd_110"],
            ["Cd",111.902757,24.13,111,"Cd_111"],
            ["Cd",112.904401,12.22,112,"Cd_112"],
            ["Cd",113.903358,28.73,113,"Cd_113"],
            ["Cd",115.904755,7.49,115,"Cd_115"],
            ["Ca",39.962591,96.941,39,"Ca_39"],
            ["Ca",41.958618,0.647,41,"Ca_41"],
            ["Ca",42.958767,0.135,42,"Ca_42"],
            ["Ca",43.955481,2.086,43,"Ca_43"],
            ["Ca",45.953693,0.004,45,"Ca_45"],
            ["Ca",47.952534,0.187,47,"Ca_47"],
            ["C",12.0,98.93,12,"C_12"],
            ["C",13.003355,1.07,13,"C_13"],
            ["Ce",135.907144,0.185,135,"Ce_135"],
            ["Ce",137.905986,0.251,137,"Ce_137"],
            ["Ce",139.905434,88.45,139,"Ce_139"],
            ["Ce",141.90924,11.114,141,"Ce_141"],
            ["Cs",132.905447,100.0,132,"Cs_132"],
            ["Cl",34.968853,75.78,34,"Cl_34"],
            ["Cl",36.965903,24.22,36,"Cl_36"],
            ["Cr",49.94605,4.345,49,"Cr_49"],
            ["Cr",51.940512,83.789,51,"Cr_51"],
            ["Cr",52.940654,9.501,52,"Cr_52"],
            ["Cr",53.938885,2.365,53,"Cr_53"],
            ["Co",58.9332,100.0,58,"Co_58"],
            ["Cu",62.929601,69.17,62,"Cu_62"],
            ["Cu",64.927794,30.83,64,"Cu_64"],
            ["Dy",155.924278,0.06,155,"Dy_155"],
            ["Dy",157.924405,0.1,157,"Dy_157"],
            ["Dy",159.925194,2.34,159,"Dy_159"],
            ["Dy",160.92693,18.91,160,"Dy_160"],
            ["Dy",161.926795,25.51,161,"Dy_161"],
            ["Dy",162.928728,24.9,162,"Dy_162"],
            ["Dy",163.929171,28.18,163,"Dy_163"],
            ["Er",161.928775,0.14,161,"Er_161"],
            ["Er",163.929197,1.61,163,"Er_163"],
            ["Er",165.93029,33.61,165,"Er_165"],
            ["Er",166.932045,22.93,166,"Er_166"],
            ["Er",167.932368,26.78,167,"Er_167"],
            ["Er",169.93546,14.93,169,"Er_169"],
            ["Eu",150.919846,47.81,150,"Eu_150"],
            ["Eu",152.921226,52.19,152,"Eu_152"],
            ["F",18.998403,100.0,18,"F_18"],
            ["Ga",68.925581,60.108,68,"Ga_68"],
            ["Ga",70.924705,39.892,70,"Ga_70"],
            ["Gd",151.919788,0.2,151,"Gd_151"],
            ["Gd",153.920862,2.18,153,"Gd_153"],
            ["Gd",154.822619,14.8,154,"Gd_154"],
            ["Gd",155.92212,20.47,155,"Gd_155"],
            ["Gd",156.923957,15.65,156,"Gd_156"],
            ["Gd",157.924101,24.84,157,"Gd_157"],
            ["Gd",159.927051,21.86,159,"Gd_159"],
            ["Ge",69.92425,20.84,69,"Ge_69"],
            ["Ge",71.922076,27.54,71,"Ge_71"],
            ["Ge",72.923459,7.73,72,"Ge_72"],
            ["Ge",73.921178,36.5,73,"Ge_73"],
            ["Ge",75.921403,7.61,75,"Ge_75"],
            ["Au",196.966552,100.0,196,"Au_196"],
            ["Hf",173.94004,0.16,173,"Hf_173"],
            ["Hf",175.941402,5.26,175,"Hf_175"],
            ["Hf",176.94322,18.6,176,"Hf_176"],
            ["Hf",177.943698,27.28,177,"Hf_177"],
            ["Hf",178.945815,13.62,178,"Hf_178"],
            ["Hf",179.946549,35.08,179,"Hf_179"],
            ["He",3.016029,0.000137,3,"He_3"],
            ["He",4.002603,99.999863,4,"He_4"],
            ["Ho",164.930319,100.0,164,"Ho_164"],
            ["H",1.007825,99.9885,1,"H_1"],
            ["H",2.014102,0.115,2,"H_2"],
            ["In",112.904061,4.29,112,"In_112"],
            ["In",114.903878,95.71,114,"In_114"],
            ["I",126.904468,100.0,126,"I_126"],
            ["Ir",190.960591,37.3,190,"Ir_190"],
            ["Ir",192.962924,62.7,192,"Ir_192"],
            ["Fe",53.939615,5.845,53,"Fe_53"],
            ["Fe",55.934942,91.754,55,"Fe_55"],
            ["Fe",56.935399,2.119,56,"Fe_56"],
            ["Fe",57.93328,0.282,57,"Fe_57"],
            ["Kr",77.920386,0.35,77,"Kr_77"],
            ["Kr",79.916378,2.28,79,"Kr_79"],
            ["Kr",81.913485,11.58,81,"Kr_81"],
            ["Kr",82.914136,11.49,82,"Kr_82"],
            ["Kr",83.911507,57.0,83,"Kr_83"],
            ["Kr",85.91061,17.3,85,"Kr_85"],
            ["La",137.907107,0.09,137,"La_137"],
            ["La",138.906348,99.91,138,"La_138"],
            ["Pb",203.973029,1.4,203,"Pb_203"],
            ["Pb",205.974449,24.1,205,"Pb_205"],
            ["Pb",206.975881,22.1,206,"Pb_206"],
            ["Pb",207.976636,52.4,207,"Pb_207"],
            ["Li",6.015122,7.59,6,"Li_6"],
            ["Li",7.016004,92.41,7,"Li_7"],
            ["Lu",174.940768,97.41,174,"Lu_174"],
            ["Lu",175.942682,2.59,175,"Lu_175"],
            ["Mg",23.985042,78.99,23,"Mg_23"],
            ["Mg",24.985837,10.0,24,"Mg_24"],
            ["Mg",25.982593,11.01,25,"Mg_25"],
            ["Mn",54.93805,100.0,54,"Mn_54"],
            ["Hg",195.965815,0.15,195,"Hg_195"],
            ["Hg",197.966752,9.97,197,"Hg_197"],
            ["Hg",198.968262,16.87,198,"Hg_198"],
            ["Hg",199.968309,23.1,199,"Hg_199"],
            ["Hg",200.970285,13.18,200,"Hg_200"],
            ["Hg",201.970626,29.86,201,"Hg_201"],
            ["Hg",203.973476,6.87,203,"Hg_203"],
            ["Mo",91.90681,14.84,91,"Mo_91"],
            ["Mo",93.905088,9.25,93,"Mo_93"],
            ["Mo",94.905841,15.92,94,"Mo_94"],
            ["Mo",95.904679,16.68,95,"Mo_95"],
            ["Mo",96.906021,9.55,96,"Mo_96"],
            ["Mo",97.905408,24.13,97,"Mo_97"],
            ["Mo",99.907477,9.63,99,"Mo_99"],
            ["Nd",141.907719,27.2,141,"Nd_141"],
            ["Nd",142.90981,12.2,142,"Nd_142"],
            ["Nd",143.910083,23.8,143,"Nd_143"],
            ["Nd",144.912569,8.3,144,"Nd_144"],
            ["Nd",145.913112,17.2,145,"Nd_145"],
            ["Nd",147.916889,5.7,147,"Nd_147"],
            ["Nd",149.920887,5.6,149,"Nd_149"],
            ["Ne",19.99244,90.48,19,"Ne_19"],
            ["Ne",20.993847,0.27,20,"Ne_20"],
            ["Ne",21.991386,9.25,21,"Ne_21"],
            ["Ni",57.935348,68.0769,57,"Ni_57"],
            ["Ni",59.930791,26.2231,59,"Ni_59"],
            ["Ni",60.93106,1.1399,60,"Ni_60"],
            ["Ni",61.928349,3.6345,61,"Ni_61"],
            ["Ni",63.92797,0.9256,63,"Ni_63"],
            ["Nb",92.906378,100.0,92,"Nb_92"],
            ["N",14.003074,99.632,14,"N_14"],
            ["N",15.000109,0.368,15,"N_15"],
            ["Os",183.952491,0.02,183,"Os_183"],
            ["Os",185.953838,1.59,185,"Os_185"],
            ["Os",186.955748,1.96,186,"Os_186"],
            ["Os",187.955836,13.24,187,"Os_187"],
            ["Os",188.958145,16.15,188,"Os_188"],
            ["Os",189.958445,26.26,189,"Os_189"],
            ["Os",191.961479,40.78,191,"Os_191"],
            ["O",15.994915,99.757,15,"O_15"],
            ["O",16.999132,0.038,16,"O_16"],
            ["O",17.99916,0.205,17,"O_17"],
            ["Pd",101.905608,1.02,101,"Pd_101"],
            ["Pd",103.904035,11.14,103,"Pd_103"],
            ["Pd",104.905084,22.33,104,"Pd_104"],
            ["Pd",105.903483,27.33,105,"Pd_105"],
            ["Pd",107.903894,26.46,107,"Pd_107"],
            ["Pd",109.905152,11.72,109,"Pd_109"],
            ["P",30.973762,100.0,30,"P_30"],
            ["Pt",189.95993,0.014,189,"Pt_189"],
            ["Pt",191.961035,0.782,191,"Pt_191"],
            ["Pt",193.962664,32.967,193,"Pt_193"],
            ["Pt",194.964774,33.832,194,"Pt_194"],
            ["Pt",195.964935,25.242,195,"Pt_195"],
            ["Pt",197.967876,7.163,197,"Pt_197"],
            ["K",38.963707,93.2581,38,"K_38"],
            ["K",39.963999,0.0117,39,"K_39"],
            ["K",40.961826,6.7302,40,"K_40"],
            ["Pr",140.907648,100.0,140,"Pr_140"],
            ["Re",184.952956,37.4,184,"Re_184"],
            ["Re",186.955751,62.6,186,"Re_186"],
            ["Rh",102.905504,100.0,102,"Rh_102"],
            ["Rb",84.911789,72.17,84,"Rb_84"],
            ["Rb",86.909183,27.83,86,"Rb_86"],
            ["Ru",95.907598,5.54,95,"Ru_95"],
            ["Ru",97.905287,1.87,97,"Ru_97"],
            ["Ru",98.905939,12.76,98,"Ru_98"],
            ["Ru",99.90422,12.6,99,"Ru_99"],
            ["Ru",100.905582,17.06,100,"Ru_100"],
            ["Ru",101.90435,31.55,101,"Ru_101"],
            ["Ru",103.90543,18.62,103,"Ru_103"],
            ["Sm",143.911995,3.07,143,"Sm_143"],
            ["Sm",146.914893,14.99,146,"Sm_146"],
            ["Sm",147.914818,11.24,147,"Sm_147"],
            ["Sm",148.91718,13.82,148,"Sm_148"],
            ["Sm",149.917271,7.38,149,"Sm_149"],
            ["Sm",151.919728,26.75,151,"Sm_151"],
            ["Sm",153.922205,22.75,153,"Sm_153"],
            ["Sc",44.95591,100.0,44,"Sc_44"],
            ["Se",73.922477,0.89,73,"Se_73"],
            ["Se",75.919214,9.37,75,"Se_75"],
            ["Se",76.919915,7.63,76,"Se_76"],
            ["Se",77.91731,23.77,77,"Se_77"],
            ["Se",79.916522,49.61,79,"Se_79"],
            ["Se",81.9167,8.73,81,"Se_81"],
            ["Si",27.976927,92.2297,27,"Si_27"],
            ["Si",28.976495,4.6832,28,"Si_28"],
            ["Si",29.97377,3.0872,29,"Si_29"],
            ["Ag",106.905093,51.839,106,"Ag_106"],
            ["Ag",108.904756,48.161,108,"Ag_108"],
            ["Na",22.98977,100.0,22,"Na_22"],
            ["Sr",83.913425,0.56,83,"Sr_83"],
            ["Sr",85.909262,9.86,85,"Sr_85"],
            ["Sr",86.908879,7.0,86,"Sr_86"],
            ["Sr",87.905614,82.58,87,"Sr_87"],
            ["S",31.972071,94.93,31,"S_31"],
            ["S",32.971458,0.76,32,"S_32"],
            ["S",33.967867,4.29,33,"S_33"],
            ["S",35.967081,0.02,35,"S_35"],
            ["Ta",179.947466,0.012,179,"Ta_179"],
            ["Ta",180.947996,99.988,180,"Ta_180"],
            ["Te",119.90402,0.09,119,"Te_119"],
            ["Te",121.903047,2.55,121,"Te_121"],
            ["Te",122.904273,0.89,122,"Te_122"],
            ["Te",123.902819,4.74,123,"Te_123"],
            ["Te",124.904425,7.07,124,"Te_124"],
            ["Te",125.903306,18.84,125,"Te_125"],
            ["Te",127.904461,31.74,127,"Te_127"],
            ["Te",129.906223,34.08,129,"Te_129"],
            ["Tb",158.925343,100.0,158,"Tb_158"],
            ["Tl",202.972329,29.524,202,"Tl_202"],
            ["Tl",204.974412,70.476,204,"Tl_204"],
            ["Th",232.03805,100.0,232,"Th_232"],
            ["Tm",168.934211,100.0,168,"Tm_168"],
            ["Sn",111.904821,0.97,111,"Sn_111"],
            ["Sn",113.902782,0.66,113,"Sn_113"],
            ["Sn",114.903346,0.34,114,"Sn_114"],
            ["Sn",115.901744,14.54,115,"Sn_115"],
            ["Sn",116.902954,7.68,116,"Sn_116"],
            ["Sn",117.901606,24.22,117,"Sn_117"],
            ["Sn",118.903309,8.59,118,"Sn_118"],
            ["Sn",119.902197,32.58,119,"Sn_119"],
            ["Sn",121.90344,4.63,121,"Sn_121"],
            ["Sn",123.905275,5.79,123,"Sn_123"],
            ["Ti",45.952629,8.25,45,"Ti_45"],
            ["Ti",46.951764,7.44,46,"Ti_46"],
            ["Ti",47.947871,73.72,47,"Ti_47"],
            ["W",179.946706,0.12,179,"W_179"],
            ["W",181.948206,26.5,181,"W_181"],
            ["W",182.950224,14.31,182,"W_182"],
            ["W",183.950933,30.64,183,"W_183"],
            ["W",185.954362,28.43,185,"W_185"],
            ["U",234.040946,0.0055,234,"U_234"],
            ["U",235.043923,0.72,235,"U_235"],
            ["U",238.050783,99.2745,238,"U_238"],
            ["V",49.947163,0.25,49,"V_49"],
            ["V",50.943964,99.75,50,"V_50"],
            ["Xe",123.905896,0.09,123,"Xe_123"],
            ["Xe",125.904269,0.09,125,"Xe_125"],
            ["Xe",127.90353,1.92,127,"Xe_127"],
            ["Xe",128.904779,26.44,128,"Xe_128"],
            ["Xe",129.903508,10.44,129,"Xe_129"],
            ["Xe",130.905082,21.18,130,"Xe_130"],
            ["Xe",131.904154,26.89,131,"Xe_131"],
            ["Xe",133.905395,10.44,133,"Xe_133"],
            ["Xe",135.90722,8.87,135,"Xe_135"],
            ["Yb",167.933894,0.13,167,"Yb_167"],
            ["Yb",169.934759,3.04,169,"Yb_169"],
            ["Yb",170.936322,14.28,170,"Yb_170"],
            ["Yb",171.936378,21.83,171,"Yb_171"],
            ["Yb",172.938207,16.13,172,"Yb_172"],
            ["Yb",173.938858,31.83,173,"Yb_173"],
            ["Yb",175.942568,12.76,175,"Yb_175"],
            ["Y",88.905848,100.0,88,"Y_88"],
            ["Zn",63.929147,48.63,63,"Zn_63"],
            ["Zn",65.926037,27.9,65,"Zn_65"],
            ["Zn",66.927131,4.1,66,"Zn_66"],
            ["Zn",67.924848,18.75,67,"Zn_67"],
            ["Zn",69.925325,0.62,69,"Zn_69"],
            ["Zr",89.904704,51.45,89,"Zr_89"],
            ["Zr",90.905645,11.22,90,"Zr_90"],
            ["Zr",91.90504,17.15,91,"Zr_91"],
            ["Zr",93.906316,17.38,93,"Zr_93"],
            ["Zr",95.908276,2.8,95,"Zr_95"]])

if __name__ == '__main__':
    print(brutto_gen())