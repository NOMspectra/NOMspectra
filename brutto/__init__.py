import pandas as pd
import re
import numpy as np
from utils import calculate_mass
from typing import Union, Mapping, Dict, Sequence, Tuple
from collections import Counter


class Brutto(object):
    def __init__(self, brutto: Union[str, Mapping[str, int]]) -> None:
        self.brutto = brutto

        if isinstance(brutto, str):
            if "_" in brutto:
                brutto = brutto.replace("_", "")

    def parse_brutto(self, brutto: str) -> Mapping[str, int]:
        """
        Ca3(PO4)2 -> {'Ca': 3, 'P': 4, 'O': 4}


        """
        br = Counter()
        pattern = "[A-Z][a-z]?\d*|\((?:[^()]*(?:\(.*\))?[^()]*)+\)\d+"
        while "(" in brutto:
            for part in re.findall(pattern, brutto):
                pass

        for part in re.findall(pattern, brutto):
            if "(" in part:
                pass
            else:
                element = re.findall("[A-Z][a-z]", part)[0]
                number = re.findall("\d+", part)

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def to_dict(self) -> Dict[str, int]:
        raise NotImplementedError()

    def to_tuple(self):
        raise NotImplementedError()

    def elemets(self):
        raise NotImplementedError()

    def exact_mass(self):
        raise NotImplementedError()

    def build_distribution(self):
        raise NotImplementedError()

    @staticmethod
    def from_array_to_bruttos() -> Sequence["Brutto"]:
        raise NotImplementedError()

    @staticmethod
    def from_bruttos_to_array() -> Tuple[Sequence[Sequence], Sequence[str]]:
        pass
