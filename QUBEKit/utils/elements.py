#!/usr/bin/env python3


class Element:
    """
    Simple class for storing element data
    """

    def __init__(self, name, number, mass):

        self.name = name
        self.number = number
        self.mass = mass


        # class PeriodicTable:
        #     """
        #     Contains full periodic table with names, and atomic numbers and masses.
        #     Has methods for retrieving mass, number, name and all 3 based on a single identifier.
        #     e.g.
        #         PeriodicTable().get_number('H') returns 1.
        #         PeriodicTable().get_atom('He') returns ('He', 2, 4.002602)
        #         etc
        #     """
        #
        #     def __init__(self):
        #
        #         self.elements = {
        #             'hydrogen': Element('H', 1, 1.00794),
        #             'helium': Element('He', 2, 4.002602),
        #             'lithium': Element('Li', 3, ),
        #             'beryllium': Element('Be', 4, ),
        #             'boron': Element('B', 5, ),
        #             'carbon': Element('C', 6, ),
        #             'nitrogen': Element('N', 7, ),
        #             'oxygen': Element('O', 8, ),
        #             'fluorine': Element('F', 9, ),
        #             'neon': Element('Ne', 10, ),
        #             'sodium': Element('Na', 11, ),
        #             'magnesium': Element('Mg', 12, ),
        #             'aluminium': Element('Al', 13, ),
        #             'silicon': Element('Si', 14, ),
        #             'phosphorus': Element('P', 15, ),
        #             'sulfur': Element('S', 16, ),
        #             'chlorine': Element('Cl', 17, ),
        #             'argon': Element('Ar', 18, ),
        #             'potassium': Element('K', 19, ),
        #             'calcium': Element('Ca', 20, ),
        #             'scandium': Element('Sc', 21, ),
        #             'titanium': Element('Ti', 22, ),
        #             'vanadium': Element('V', 23, ),
        #             'chromium': Element('Cr', 24, ),
        #             'manganese': Element('Mn', 25, ),
        #             'iron': Element('Fe', 26, ),
        #             'cobalt': Element('Co', 27, ),
        #             'nickel': Element('Ni', 28, ),
        #             'copper': Element('',, ),
        #             'zinc': Element('',, ),
        #             'gallium': Element('',, ),
        #             'germanium': Element('',, ),
        #             'arsenic': Element('',, ),
        #             'selenium': Element('',, ),
        #             'bromium': Element('',, ),
        #             'krypton': Element('',, ),
        #             'rubidium': Element('',, ),
        #             'strontium': Element('',, ),
        #             'yttrium': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #             '': Element('',, ),
        #
        # }

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
