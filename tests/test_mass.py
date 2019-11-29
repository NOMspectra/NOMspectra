import logging
import os
import time
import unittest

import numpy as np
import pandas as pd

from brutto_generator import generate_brutto_formulas
from mass import MassSpectrum
from utils import calculate_mass
import settings
from functools import reduce


class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.ms = MassSpectrum()
        self.ms.load("test.csv")
        self.ms = self.ms.drop_unassigned()

        self.masses = []

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        for filename in os.listdir(settings.DATA_FOLDER):
            self.masses.append(MassSpectrum())
            self.masses[-1].load(
                f"{settings.DATA_FOLDER}/{filename}",
                mapper=mapper,
                sep=',',
                ignore_columns=["peakNo", "errorPPM", "DBE", "class", "z"]
            )

    def test_get_mass(self):

        bruttos = [
            (1, 0, 1, 0, 0),  # CO
            (0, 0, 0, 2, 0),  # N2
            (2, 4, 0, 0, 0),  # C2H4
            (1, 4, 0, 0, 0),  # CH4
            (1, 5, 0, 1, 0),  # CH3NH2
            (1, 4, 0, 0, 1),  # CH3SH
        ]

        ans = calculate_mass(bruttos)
        self.assertEqual(len(ans), 6)

        expected = [27.994915, 28.006148, 28.0313, 16.0313, 31.042199, 48.003371]
        for i in range(6):
            self.assertAlmostEqual(expected[i], ans[i], delta=0.0001)

    def test_generate_brutto_formulas(self):
        df = generate_brutto_formulas()

        masses = df["mass"].values
        self.assertTrue(all(masses[i] <= masses[i+1] for i in range(len(masses)-1)))

    def test_load(self):
        ms = MassSpectrum()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load(f"{settings.DATA_FOLDER}/a_1.csv", mapper, sep=",")  # FIXME relative paths are bad

        self.assertTrue("mass" in ms.table)
        self.assertTrue("I" in ms.table)

        self.assertFalse("mw" in ms.table)
        self.assertFalse("relativeAbundance" in ms.table)

    def test_mass_spectrum_constructor(self):
        ms = MassSpectrum()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load(f"{settings.DATA_FOLDER}/a_1.csv", mapper, sep=",")

        ms = MassSpectrum(ms.table)

    def test_assign(self):

        # load a spectrum
        ms = MassSpectrum()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load(
            "../data/a_1.csv",
            mapper,
            sep=',',
            ignore_columns=["peakNo", "errorPPM", "DBE", "class", "C", "H", "O", "N", "S", "z"]
        )

        gen_brutto = pd.read_csv("../brutto_generator/C_H_O_N_S.csv", sep=";")

        T = time.time()
        ms = ms.assign(gen_brutto, elems=list("CHONS"))

        self.logger.info(f"Spectrum assignment is done for {time.time() - T} seconds")

    def test_assign_only_masses(self):

        N = 20
        masses = (np.random.rand(N, 1) + 0.8) / 2 * 1000

        df = pd.DataFrame(masses, columns=["mass"])

        ms = MassSpectrum(df)

        gen_brutto = pd.read_csv("../brutto_generator/C_H_O_N_S.csv", sep=";")
        ms = ms.assign(gen_brutto, elems=list("CHNOS"))

        self.assertEqual(N, len(ms))

        self.assertTrue("C" in ms.table)
        self.assertTrue("H" in ms.table)
        self.assertTrue("N" in ms.table)
        self.assertTrue("O" in ms.table)
        self.assertTrue("S" in ms.table)

    def test_xor(self):
        # load a spectrum
        ms = self.ms.drop_unassigned()

        ms = ms ^ ms

        self.assertEqual(len(ms), 0)

    def test_and(self):
        ms = self.ms.drop_unassigned()
        length = len(ms)
        ms = ms & ms
        self.assertEqual(length, len(ms))

        self.assertEqual(1, np.mean(ms.table.numbers == 2))

    def test_or_equal(self):
        ms = self.ms.drop_unassigned()
        length = len(ms)
        ms = ms | ms

        self.assertEqual(length, len(ms))

        self.assertEqual(1, np.mean(ms.table.numbers == 2))

    def test_or_with_empty(self):
        ms_1 = self.ms.drop_unassigned()
        length = len(ms_1)

        ms_2 = MassSpectrum()

        ms = ms_1 | ms_2

        self.assertEqual(length, len(ms))
        self.assertEqual(1, np.mean(ms.table.numbers == 1))

    def test_get_brutto_dict(self):
        ms_1 = self.ms
        ms_2 = MassSpectrum()

        ms_1.get_brutto_dict()

        ms_2.get_brutto_dict()

    def test_and_with_empties(self):
        ms = MassSpectrum()

        ms = ms & ms

        self.assertEqual(len(ms), 0)

    def test_and_with_one_empty(self):
        ms_1 = MassSpectrum()
        ms_2 = self.ms

        ms = ms_1 & ms_2

        self.assertEqual(len(ms), 0)

    def test_xor_with_one_empty(self):
        ms_1 = MassSpectrum()
        ms_2 = self.ms

        ms = ms_1 ^ ms_2

        self.assertEqual(len(ms), len(ms_2))

    def test_sum_of_ms(self):
        ms_1 = self.ms
        ms_2 = MassSpectrum()

        ms = ms_1 + ms_2 + ms_1 + ms_2

        self.assertEqual(1, np.mean(ms.table.numbers == 2))

    def test_subtract(self):
        ms = self.ms

        ms = ms - ms

        self.assertEqual(0, len(ms))

    def test_subtract_with_empty(self):
        ms_1 = self.ms
        ms_2 = MassSpectrum()

        self.assertEqual(0, len(ms_2 - ms_1))
        self.assertEqual(len(ms_1), len(ms_1 - ms_2))

    def test_save(self):
        ms = self.ms

        ms.save("tmp.csv")

        os.remove("tmp.csv")

    def test_arithmetic(self):
        x, y, z = self.masses[:3]

        union = x + y + z

        lens = []
        for i in [x, y, z]:
            lens.append(len(i))
            union.calculate_jaccard_needham_score(i)

        self.assertTrue(len(union > 2) > 0)

        x -= union
        y -= union
        z -= union

        for i, ms in enumerate([x, y, z]):
            self.assertTrue(len(ms) < lens[i])

    def test_calculate_dbe(self):
        pass


if __name__ == "__main__":
    unittest.main()
