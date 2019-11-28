import unittest
from utils import calculate_mass
from brutto_generator import generate_brutto_formulas
from mass import MassSpectra


class Test(unittest.TestCase):

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
        print(ans)
        self.assertEqual(len(ans), 6)

        expected = [27.994915, 28.006148, 28.0313, 16.0313, 31.042199, 48.003371]
        for i in range(6):
            self.assertAlmostEqual(expected[i], ans[i], delta=0.0001)

    def test_generate_brutto_formulas(self):
        df = generate_brutto_formulas()

        masses = df["mass"].values
        self.assertTrue(all(masses[i] <= masses[i+1] for i in range(len(masses)-1)))

    def test_load(self):
        ms = MassSpectra()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load("../data/CHA-Florida.csv", mapper, sep=",")  # FIXME relative paths are bad

        self.assertTrue("mass" in ms.table)
        self.assertTrue("I" in ms.table)

        self.assertFalse("mw" in ms.table)
        self.assertFalse("relativeAbundance" in ms.table)

    def test_mass_spectra_constructor(self):
        ms = MassSpectra()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load("../data/CHA-Florida.csv", mapper, sep=",")  # FIXME relative paths are bad

        ms = MassSpectra(ms.table)

    def test_assign(self):
        ms = MassSpectra()

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        ms.load("../data/CHA-Florida.csv", mapper)

        ms = ms.assign

    def test_save(self):
        pass

    def test_calculate_dbe(self):
        pass


if __name__ == "__main__":
    unittest.main()
