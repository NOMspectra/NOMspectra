from mass import MassSpectrum


def _calculate_tmds_by_default(mass_spectrum: MassSpectrum) -> MassSpectrum:
    """Calculates tmds spectrum by honest N^2 algorithm

    :param mass_spectrum:
    :return:
    """
    raise NotImplementedError


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