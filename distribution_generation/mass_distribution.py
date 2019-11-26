import logging
import time
from typing import Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import multinomial

import settings

logger = logging.getLogger(__name__)


class IsotopeDistribution(object):
    masses = pd.read_csv(settings.ISOTOPE_ABUNDANCE_PATH, sep=";")

    def __init__(self, brutto: Union[dict, str]) -> None:
        if isinstance(brutto, str):
            raise NotImplementedError("parsing of string is not Implemented")

        self.brutto = brutto  # {"C": 2, "H": 6, "O": 1}
        self.generated_masses = np.array([])

    def generate_iterations(self, n: int):
        """
        Draw a sample of the molecule N times
        :param n: number of iterations
        :return: None
        """

        res = np.zeros((n, 0))
        for elem in self.brutto:
            df = self.masses[self.masses.element == elem][["mass", "abundance"]]
            df["abundance"] = df["abundance"] / df["abundance"].sum()
            m = df["mass"].values * multinomial(self.brutto[elem], df["abundance"], size=n)

            res = np.concatenate([res, m], axis=1)

        self.generated_masses = np.concatenate((self.generated_masses, res.sum(axis=1)))
        self.generated_masses.sort()

    def get_density(self, x: Sequence[float], eps: float = 0.05) -> Sequence[float]:
        """
        Calculates density for each x, uses self.generated_masses for estimation
        :param x: masses for which the function calculates density
        :param eps:
        :return: array of densities for each mass
        """

        # TODO probably not the most effective way to calculate density
        # TODO replace linear sum of to more statistic way

        size = len(x)
        y = np.zeros(size)
        for i in range(size):
            left = np.searchsorted(self.generated_masses, x[i] - eps)
            right = np.searchsorted(self.generated_masses, x[i] + eps)
            y[i] = right - left

        return y

    def draw(self, eps=0.05):
        x = np.linspace(self.generated_masses.min() - 1, self.generated_masses.max() + 1, 1000)
        y = self.get_density(x, eps=eps)

        plt.figure(figsize=(10, 4))
        plt.xlabel("Mass, Da")
        plt.ylabel("Abundance")
        plt.title(f"Mass Distribution for {''.join([''.join((i, str(self.brutto[i]))) for i in self.brutto])}")
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    # d = IsotopeDistribution({"C": 2, "H": 6, "O": 1})
    brutto = {"Na": 2, "Pt": 3, "Cl": 4}
    T = time.time()
    d = IsotopeDistribution(brutto)
    d.generate_iterations(100000)
    d.draw()
    print("Time:", time.time() - T, "seconds")

