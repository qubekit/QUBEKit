#!/usr/bin/env python3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class Engines:
    """
    Engines base class containing core information that all other engines
    (PSI4, Gaussian etc) will have.
    """

    def __init__(self, molecule: "Ligand"):
        self.molecule = molecule

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"
