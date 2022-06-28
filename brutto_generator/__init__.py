import time
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

def brutto_gen(elems = {'C':(1, 60),'H':(0,100),'O':(0,60), 'N':(0,3), 'S':(0,2)},
               masses_path='masses/element_table.csv'):
    """Generete brutto formulas
    :param elems: dictonary contains element:his_range. 
    Not main isotopes mark as El_x, where El - element, x - its mass
    Examples
    'C':(1,60) - content of carbon (main isotope) from 1 to 59 (60-1)
    'O_18':(0,3) - conent of isotope 18 oxygen from 0 to 2
    """

    #load elements table. Generatete in mass folder
    elems_mass_table = pd.read_csv(masses_path)
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
    if 'H' in gdf.columns and 'C' in gdf.columns:
        gdf['H/C'] = gdf['H']/gdf['C']
        gdf['O/C'] = gdf['O']/gdf['C']
        gdf = gdf.loc[(gdf['H/C'] < 2.2) & (gdf['H/C'] > 0.25) & (gdf['O/C'] < 1)]
        gdf = gdf.drop(columns=['H/C','O/C'])
    
    if 'N' in gdf.columns and 'H' in gdf.columns:
        gdf['parity'] = (gdf['H'] + gdf['N'])%2
        gdf = gdf.loc[gdf['parity']==0]
        gdf = gdf.drop(columns=['parity'])

    #calculate mass
    masses = np.array(list(elems_dict.values()))
    gdf['mass'] = gdf.multiply(masses).sum(axis=1)

    gdf = gdf.sort_values("mass").reset_index(drop=True)

    return gdf

def brutto_gen_dummy(
    elems: str = 'CHONS',
    round_search: int = 5,
    C = [1, 60],
    H = [0, 100],
    O = [0, 60],
    N = [0, 3],
    S = [0, 2]
):
    """Generates brutto formula 
    :param elems: CHO - find just CHO formules, CHON - add nitrigen to find, CHONS - add sulfur
    :param round: default 5, round of caclulated mass from brutto by this
    :return: dict {mass : [C, H, O, N, S], ...}
    """

    H_1 = 1.007825
    C_12 = 12.000000
    N_14 = 14.003074
    O_16 = 15.994915
    S_32 = 31.972071

    brutto_gen = {}
    for c in range(C[0], C[1]):
        for h in range (H[0], H[1]):
            for o in range (O[0], O[1]):
                for n in range (N[0], N[1]):
                    for s in range (S[0], S[1]):
                        
                        # compounds with these atomic ratios are unlikely to meet
                        if h/c > 2.2 or h/c < 0.25:
                            continue
                        if o/c > 1:
                            continue
                        
                        #check hydrogen atom parity
                        if (h%2 == 0 and n%2 != 0) or (n%2 == 0 and h%2 != 0):
                            continue

                        if n > 0 and 'N' not in elems:
                            continue
                        if s > 0 and 'S' not in elems:
                            continue
                        
                        mass = round(c*C_12 + h*H_1 + o*O_16 + n*N_14 + s*S_32, round_search)
                        brutto_gen[mass] = [c, h, o, n, s]

    return brutto_gen

def get_dataframe(gen):
    return pd.DataFrame([{
        "mass": i,
        "C": gen[i][0],
        "H": gen[i][1],
        "O": gen[i][2],
        "N": gen[i][3],
        "S": gen[i][4],
    } for i in gen])

def get_dummy_gdf(elems: str = 'CHONS',
            round_search: int = 5,
            C = [1, 60],
            H = [0, 100],
            O = [0, 60],
            N = [0, 3],
            S = [0, 2]):
    generated_bruttos_table = brutto_gen_dummy(elems=elems,
                                                round_search=round_search,
                                                C=C,
                                                H=H,
                                                O=O,
                                                N=N,
                                                S=S)
    gdf = get_dataframe(generated_bruttos_table)
    gdf = gdf.sort_values("mass")
    return(gdf)


if __name__ == '__main__':
    T = time.time()
    gdf = brutto_gen(masses_path='element_table.csv')
    print(len(gdf))
    print(time.time() - T)