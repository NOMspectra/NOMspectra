import unittest
from mass import calculate_mass, generate_brutto_formulas


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


if __name__ == "__main__":
    unittest.main()
