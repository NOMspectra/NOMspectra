import time
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
import pickle


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
                        if h/c > 2 or h/c < 0.25:
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

def get_gdf(elems: str = 'CHONS',
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

def save_gdf(gdf, path='brutto_generator/gdf.pickle'):
    with open(path, 'wb') as f:
        pickle.dump(gdf, f)

def load_gdf(path='brutto_generator/gdf.pickle'):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    T = time.time()
    gdf = get_gdf()
    save_gdf(gdf, path='gdf.pickle')
    print(time.time() - T)