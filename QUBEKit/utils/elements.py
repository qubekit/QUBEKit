#!/usr/bin/env python3


class Element:
    """
    Simple class for storing element data
    """

    def __init__(self, name, number, mass):

        self.name = name
        self.number = number
        self.mass = mass


class PeriodicTable:
    """
    Contains full periodic table with names, and atomic numbers and masses.
    Has methods for retrieving mass, number, name and all 3 based on a single identifier.
    e.g.
        PeriodicTable().get_number('H') returns 1.
        PeriodicTable().get_atom('He') returns ('He', 2, 4.002602)
        etc
    """

    def __init__(self):

        self.elements = {'hydrogen': Element('H', 1, 1.00794),
                         'helium': Element('He', 2, 4.002602)}

    def get_mass(self, identifier):
        """Given a name or atomic number, return the mass of the element."""
        for ele in self.elements.values():
            if type(identifier) is str:
                if ele.name == identifier:
                    return ele.mass
            elif type(identifier) is int:
                if ele.number == identifier:
                    return ele.mass

    def get_number(self, identifier):
        """Given a name or atomic mass, return the atomic number of the element."""
        for ele in self.elements.values():
            if type(identifier) is str:
                if ele.name == identifier:
                    return ele.number
            elif type(identifier) is float:
                if ele.mass == identifier:
                    return ele.number

    def get_name(self, identifier):
        """Given an atomic number or mass, return the name of the element."""
        for ele in self.elements.values():
            if type(identifier) is int:
                if ele.number == identifier:
                    return ele.name
            elif type(identifier) is float:
                if ele.mass == identifier:
                    return ele.name

    def get_atom(self, identifier):
        """Given a name, atomic number or mass, return the name, atomic number and mass of the element."""
        for ele in self.elements.values():
            if type(identifier) is int:
                if ele.number == identifier:
                    return ele.name, ele.number, ele.mass
            elif type(identifier) is str:
                if ele.name == identifier:
                    return ele.name, ele.number, ele.mass
            elif type(identifier) is float:
                if ele.mass == identifier:
                    return ele.name, ele.number, ele.mass
