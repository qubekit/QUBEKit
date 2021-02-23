#!/usr/bin/env python3

from types import SimpleNamespace
from typing import List, Optional

from rdkit.Chem.rdchem import GetPeriodicTable, PeriodicTable


class CustomNamespace(SimpleNamespace):
    """
    Adds iteration and dict-style access of keys, values and items to SimpleNamespace.
    TODO Add get() method? (similar to dict)
    """

    def keys(self):
        for key in self.__dict__:
            yield key

    def values(self):
        for value in self.__dict__.values():
            yield value

    def items(self):
        for key, value in self.__dict__.items():
            yield key, value

    def __iter__(self):
        return self.items()


class Element:
    """
    Simple wrapper class for getting element info using RDKit.
    """

    @staticmethod
    def p_table() -> PeriodicTable:
        return GetPeriodicTable()

    @staticmethod
    def mass(identifier):
        pt = Element.p_table()
        return pt.GetAtomicWeight(identifier)

    @staticmethod
    def number(identifier):
        pt = Element.p_table()
        return pt.GetAtomicNumber(identifier)

    @staticmethod
    def name(identifier):
        pt = Element.p_table()
        return pt.GetElementSymbol(identifier)


class Atom:
    """
    Class to hold all of the "per atom" information.
    All atoms in Molecule will have an instance of this Atom class to describe their properties.
    """

    def __init__(
        self,
        atomic_number: int,
        atom_index: int,
        atom_name: str = "",
        partial_charge: Optional[float] = None,
        formal_charge: Optional[int] = None,
    ):

        self.atomic_number: int = atomic_number
        # The QUBEKit assigned name derived from the atomic name and its index e.g. C1, F8, etc
        self.atom_name: str = atom_name
        self.atom_index: int = atom_index
        self.partial_charge: Optional[float] = partial_charge
        self.formal_charge: Optional[int] = formal_charge
        self.atom_type: Optional[str] = None
        self.bonds: List[int] = []

    @property
    def atomic_mass(self) -> float:
        """Convert the atomic number to mass."""
        return Element.mass(self.atomic_number)

    @property
    def atomic_symbol(self) -> str:
        """Convert the atomic number to the atomic symbol as per the periodic table."""
        return Element.name(self.atomic_number).title()

    def add_bond(self, bonded_index) -> None:
        """
        Add a bond to the atom, this will make sure the bond has not already been described
        :param bonded_index: The index of the atom bonded to self
        """

        if bonded_index not in self.bonds:
            self.bonds.append(bonded_index)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self):
        """
        Prints the Atom class objects' names and values one after another with new lines between each.
        """

        return_str = ""
        for key, val in self.__dict__.items():
            # Return all objects as {atom object name} = {atom object value(s)}.
            return_str += f"\n{key} = {val}\n"

        return return_str


class ExtraSite:
    """
    Used to store extra sites for xml writer in ligand.
    This class is used by both internal v-sites fitting and the ONETEP reader.
    """

    def __init__(self):
        self.parent_index: Optional[int] = None
        self.closest_a_index: Optional[int] = None
        self.closest_b_index: Optional[int] = None
        # Optional: Used for Nitrogen only.
        self.closest_c_index: Optional[int] = None

        self.o_weights: Optional[List[float]] = None
        self.x_weights: Optional[List[float]] = None
        self.y_weights: Optional[List[float]] = None

        self.p1: Optional[float] = None
        self.p2: Optional[float] = None
        self.p3: Optional[float] = None
        self.charge: Optional[float] = None
