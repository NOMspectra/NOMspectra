import numpy as np
import pandas as pd
from typing import Sequence

from mass import MassSpectrum


def _make_smooth(X: np.ndarray) -> np.ndarray:
    return X


def _calculate_tmds_by_default(mass_spectrum: MassSpectrum) -> MassSpectrum:
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

    # FIXME here we need to smooth
    table["I"] = table["probability"]
    return MassSpectrum(table[["mass", "I"]])


def _calculate_tmds_by_stochastic(mass_spectrum: MassSpectrum) -> MassSpectrum:
    """Calculates tmds spectrum by random sampling

    :param mass_spectrum:
    :return:
    """
    raise NotImplementedError


def calculate_tmds(mass_spectrum: MassSpectrum, algo: str) -> MassSpectrum:
    """For spectrum calculates tmds spectrum

    This function is a facade
    :param mass_spectrum: Source Mass Spectrum
    :param algo: default or stochastic
    :return: tmds Mass Spectrum
    """

    if algo == "default":
        return _calculate_tmds_by_default(mass_spectrum)

    if algo == "stochastic":
        return _calculate_tmds_by_stochastic()
