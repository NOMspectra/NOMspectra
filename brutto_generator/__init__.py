import pandas as pd
from typing import Sequence
from itertools import product
import numpy as np
from utils import calculate_mass
import time

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


if __name__ == '__main__':
    T = time.time()
    df = generate_brutto_formulas(
        min_n=(6, 6, 0, 0, 0),
        max_n=(40, 40, 5, 5, 5),
        elems=tuple("CHONS")
    )
    df.to_csv("C_H_O_N_S.csv", sep=";", index=False)
    print(time.time() - T)
