import unittest

from utils import calculate_mass, NoSuchChemicalElement


class Test(unittest.TestCase):
    def test_calculate_mass_with_monoisotopic_masses(self):
        elements = "CHO"
        table = [
            (2, 6, 1),
            (0, 2, 1),
            (1, 4, 0)
        ]

        res = calculate_mass(table, elements)
        self.assertEqual(res[0], 46.041865)
        self.assertEqual(res[1], 18.010565)
        self.assertEqual(res[2], 16.0313)
        self.assertEqual(len(res), 3)

    def test_calculate_mass_with_isotopic_masses(self):
        elements = ["2H", "13C"]
        table = [
            (4, 1)
        ]
        res = calculate_mass(table, elements)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0], 21.059763)

    def test_calculate_mass_with_mix(self):
        elements = ["C", "13C", "H", "O"]
        table = [
            (2, 0, 6, 1),
            (1, 1, 6, 1),
            (0, 1, 4, 0)
        ]

        res = calculate_mass(table, elements)
        self.assertEqual(res[0], 46.041865)
        self.assertEqual(res[1], 47.04522)
        self.assertEqual(res[2], 17.034655)
        self.assertEqual(len(res), 3)

    def test_calculate_mass_bad_elements(self):
        elements = ["150C"]
        table = [(1, )]
        self.assertRaises(NoSuchChemicalElement, calculate_mass, table, elements)

        elements = ["Qw"]
        table = [(1,)]
        self.assertRaises(NoSuchChemicalElement, calculate_mass, table, elements)


if __name__ == "__main__":
    unittest.main()
