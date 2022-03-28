import time
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd

from utils import calculate_mass


def generate_brutto_formulas(
        min_n: Sequence[int] = (-10, -30, -5, -5, -0),
        max_n: Sequence[int] = (10, 30, 5, 5, 0),
        elems: Sequence[str] = tuple("CHONS")
):
    """Generates brutto formula by limit conditions and calculate masses

    :param min_n:
    :param max_n:
    :param elems: Iterable object of elements i.e "CHO" or ["Cu", "K", "C"]
    :return:
    """

    # generate brutto
    brutto = np.array(list(product(*[range(i, j + 1) for (i, j) in zip(min_n, max_n)])))

    # calculate masses
    masses = calculate_mass(brutto)

    # create pandas table for collect results
    df = pd.DataFrame()
    df["mass"] = masses

    for i, elem in enumerate(elems):
        df[elem] = brutto[:, i]

    # sorting table
    df = df.sort_values(by=["mass"])

    return df


def brutto_gen_dummy(
    elems: str = 'CHONS',
    round_search: int = 5
):
    """Generates brutto formula 
    :param elems: CHO - find just CHO formules, CHON - add nitrigen to find, CHONS - add sulfur
    :param round: default 5, round of caclulated mass from brutto by this
    :return: dict {mass : [C, H, O, N, S], ...}
    """

    H = 1.007825
    C = 12.000000
    N = 14.003074
    O = 15.994915
    S = 31.972071

    brutto_gen = {}
    for c in range(1,120,1):
        for h in range (0, 200,1):
            for o in range (0, 120,1):
                for n in range (0, 4,1):
                    for s in range (0, 2, 1):
                        
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
                        
                        mass = round(c*C + h*H + o*O + n*N + s*S, round_search)
                        brutto_gen[mass] = [c, h, o, n, s]

    return brutto_gen


if __name__ == '__main__':
    T = time.time()
    df = generate_brutto_formulas(
        min_n=(6, 6, 0, 0, 0),
        max_n=(40, 40, 5, 5, 5),
        elems=tuple("CHONS")
    )
    df.to_csv("C_H_O_N_S.csv", sep=";", index=False)
    print(time.time() - T)