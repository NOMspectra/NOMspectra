from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import multinomial

from mass import MassSpectrum


def _make_smooth(mass_spectrum: MassSpectrum) -> MassSpectrum:
    # TODO smoothing, very important!!
    # I think it should be by implemented by signals
    # and maybe this can become a MassSpectrum method

    return mass_spectrum


def _calculate_tmds_by_default(mass_spectrum: MassSpectrum, max_mass = Optional[float]) -> MassSpectrum:
    """Calculates tmds spectrum by honest N^2 algorithm

    :param mass_spectrum:
    :return:
    """

    mass_spectrum = mass_spectrum.copy()
    mass_spectrum.table["I"] = mass_spectrum.table["I"] / mass_spectrum.table["I"].sum()

    # this 2 lines makes all calculations, pure magic :)
    x = mass_spectrum[["mass", "I"]].values
    table = pd.DataFrame(
        np.hstack([np.tile(x.T, len(x)).T, np.repeat(x, len(x), axis=0)]),
        columns=["mass_1", "probability_1", "mass_2", "probability_2"]
    )
    table["mass"] = table["mass_1"] - table["mass_2"]
    table["probability"] = table["probability_1"] * table["probability_2"]

    # filtering, only masses > 0
    table = table[table.mass > 0]
    if max_mass is not None:
        table = table[table.mass < max_mass]

    table["I"] = table["probability"] / table["probability"].sum()
    return _make_smooth(MassSpectrum(table[["mass", "I"]]))


def _calculate_tmds_by_stochastic(
        mass_spectrum: MassSpectrum,
        max_mass: Optional[float],
        iterations_number: int
) -> MassSpectrum:
    """Calculates tmds spectrum by random sampling

    :param mass_spectrum:
    :return:
    """

    mass_spectrum = mass_spectrum.copy()
    mass_spectrum.table["I"] = mass_spectrum.table["I"] / mass_spectrum.table["I"].sum()

    idx_1 = np.random.choice(
        np.arange(len(mass_spectrum)),
        p=mass_spectrum.table["I"],
        size=iterations_number
    )

    idx_2 = np.random.choice(
        np.arange(len(mass_spectrum)),
        p=mass_spectrum.table["I"],
        size=iterations_number
    )

    mass = mass_spectrum["mass"]

    table = pd.DataFrame(np.array([mass[idx_1], mass[idx_2]]).T, columns=["mass_1", "mass_2"])
    table["mass"] = np.abs(table["mass_1"] - table["mass_2"])
    table["probability"] = 1.

    # filtering zeros and bif masses
    table = table[table.mass > 0]
    if max_mass is not None:
        table = table[table.mass < max_mass]

    table["I"] = table["probability"] / table["probability"].sum()
    return _make_smooth(MassSpectrum(table[["mass", "I"]]))


def calculate_tmds(
        mass_spectrum: MassSpectrum,
        algo: str = "default",
        max_mass: Optional[float] = None,
        iterations_number: Optional[int] = None
) -> MassSpectrum:
    """For spectrum calculates tmds spectrum

    This function is a facade
    :param mass_spectrum: Source Mass Spectrum
    :param algo: default or stochastic
    :param max_mass: mass limit
    :param iterations_number: if algo stochastic it's number of iterations
    :return: tmds Mass Spectrum
    """

    if algo == "default":
        return _calculate_tmds_by_default(mass_spectrum, max_mass=max_mass)

    if algo == "stochastic":
        if iterations_number is None:
            iterations_number = 10000000  # 1e8
        return _calculate_tmds_by_stochastic(mass_spectrum, max_mass=max_mass, iterations_number=iterations_number)
