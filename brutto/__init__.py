from typing import Union, Mapping, Dict, Tuple

from brutto.node import build_brutto_tree
from distribution_generation.mass_distribution import IsotopeDistribution
from utils import calculate_mass


class Brutto(object):
    def __init__(self, brutto: Union[str, Mapping[str, int]]) -> None:

        if isinstance(brutto, str):
            self.brutto = brutto
            self.parse_brutto()
        elif isinstance(brutto, dict):
            self.dict = brutto
            self.compile_brutto()

    def parse_brutto(self) -> None:
        """Takes self.brutto string and create dict of elements
        Ca3(PO4)2 -> {'Ca': 3, 'P': 4, 'O': 4}"""

        self.dict = {}
        node = build_brutto_tree(self.brutto)
        self.dict = dict(node.get_dict())

    def compile_brutto(self) -> None:
        """Takes the self.dict and make it string brutto"""

        # future brutto
        s = ""
        for element in self.dict:
            s += element + ("" if self.dict[element] == 1 else str(self.dict[element]))

        self.brutto = s

    def __str__(self):
        return self.brutto

    def __repr__(self):
        return self.brutto

    def to_dict(self) -> Dict[str, int]:
        return self.dict

    def get_coef(self) -> Tuple[int]:
        return tuple(self.dict.values())

    def get_elemets(self) -> Tuple[str]:
        return tuple(self.dict.keys())

    def exact_mass(self) -> float:
        """Calculates exact monoisotopic mass for brutto"""

        return float(calculate_mass([self.to_tuple()], elems=self.elemets())[0])

    def build_distribution(self, n: int = 100000) -> None:
        IsotopeDistribution(self.dict).generate_iterations(n).draw()


if __name__ == '__main__':
    Brutto('PtCl2').build_distribution()
