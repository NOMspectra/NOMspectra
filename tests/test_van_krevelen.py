import logging
import os
import unittest

from mass import MassSpectrum, VanKrevelen


class Test(unittest.TestCase):

    def setUp(self) -> None:
        self.logger = logging.getLogger(__name__)

        mapper = {"mw": "mass", "relativeAbundance": "I"}
        self.ms = MassSpectrum().load(
            "assigned.csv",
            mapper=mapper,
            sep=',',
            ignore_columns=["peakNo", "errorPPM", "DBE", "class", "z"],
        )

        self.masses = []

    def tearDown(self) -> None:
        vk_name = "test_saved_van_Krevelen.csv"
        figure_name = "figure.png"

        if os.path.exists(vk_name):
            os.system(f"rm {vk_name}")

        if os.path.exists(figure_name):
            os.system(f"rm {figure_name}")

    def test_draw_scatter(self) -> None:
        vk = VanKrevelen(self.ms)
        vk.draw_scatter()
        vk.show()

    def test_draw_scatter_with_marginals(self) -> None:
        vk = VanKrevelen(self.ms)
        vk.draw_scatter_with_marginals()
        vk.show()

    def test_draw_density(self) -> None:
        vk = VanKrevelen(self.ms)
        vk.draw_density()
        vk.show()

    def test_draw_density_with_marginals(self) -> None:
        vk = VanKrevelen(self.ms)
        vk.draw_density_with_marginals()
        vk.show()

    def test_save(self) -> None:
        vk = VanKrevelen(self.ms)

        filename = "test_saved_van_Krevelen.csv"
        self.assertFalse(os.path.exists("test_saved_van_Krevelen.csv"))
        vk.save(filename)
        self.assertTrue(os.path.exists("test_saved_van_Krevelen.csv"))

    def test_save_fig(self) -> None:
        vk = VanKrevelen(self.ms)
        vk.draw_density()
        vk.save_fig("figure.png", dpi=300)
        self.assertTrue(os.path.exists('figure.png'))

    def test_load(self) -> None:
        vk = VanKrevelen(self.ms)
        filename = "test_saved_van_Krevelen.csv"
        vk.save(filename)

        vk = VanKrevelen.load(filename)
        self.assertEqual(vk.name, "test_saved_van_Krevelen")


if __name__ == "__main__":
    unittest.main()
