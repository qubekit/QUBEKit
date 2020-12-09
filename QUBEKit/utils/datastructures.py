#!/usr/bin/env python3

from types import SimpleNamespace

from rdkit.Chem.rdchem import GetPeriodicTable


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

    pt = GetPeriodicTable()

    def mass(self, identifier):
        return self.pt.GetAtomicWeight(identifier)

    def number(self, identifier):
        return self.pt.GetAtomicNumber(identifier)

    def name(self, identifier):
        return self.pt.GetElementSymbol(identifier)


class Atom:
    """
    Class to hold all of the "per atom" information.
    All atoms in Molecule will have an instance of this Atom class to describe their properties.
    """

    def __init__(self, atomic_number, atom_index, atom_name='', partial_charge=None, formal_charge=None):

        self.atomic_number = atomic_number
        self.atomic_mass = Element().mass(atomic_number)
        # The actual atomic symbol as per periodic table e.g. C, F, Pb, etc
        self.atomic_symbol = Element().name(atomic_number).title()
        # The QUBEKit assigned name derived from the atomic name and its index e.g. C1, F8, etc
        self.atom_name = atom_name
        self.atom_index = atom_index
        self.partial_charge = partial_charge
        self.formal_charge = formal_charge
        self.atom_type = None
        self.bonds = []

    def add_bond(self, bonded_index):
        """
        Add a bond to the atom, this will make sure the bond has not already been described
        :param bonded_index: The index of the atom bonded to self
        """

        if bonded_index not in self.bonds:
            self.bonds.append(bonded_index)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def __str__(self):
        """
        Prints the Atom class objects' names and values one after another with new lines between each.
        """

        return_str = ''
        for key, val in self.__dict__.items():
            # Return all objects as {atom object name} = {atom object value(s)}.
            return_str += f'\n{key} = {val}\n'

        return return_str


class ExtraSite:
    """Used to store extra sites for later xml printing."""
    def __init__(self):
        self.parent_index = None        # int
        self.closest_a_index = None     # int
        self.closest_b_index = None     # int
        # Optional: Used for Nitrogen only.
        self.closest_c_index = None     # int
        self.o_weights = None           # list of float
        self.x_weights = None           # list of float
        self.y_weights = None           # list of float

        self.p1 = None                  # float
        self.p2 = None                  # float
        self.p3 = None                  # float
        self.charge = None              # float
