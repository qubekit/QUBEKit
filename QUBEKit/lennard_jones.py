#!/usr/bin/env python3

import math
from collections import namedtuple
from typing import Dict, List, Optional

from QUBEKit.ligand import Ligand
from QUBEKit.utils import constants


class LennardJones:

    # Beware weird units, (wrong in the paper too).
    # Units: vfree: Bohr ** 3, bfree: Ha * (Bohr ** 6), rfree: Angs
    FreeParams = namedtuple("params", "vfree bfree rfree")
    elem_dict: Dict[str: FreeParams] = {
        "H": FreeParams(7.6, 6.5, 1.64),
        "X": FreeParams(7.6, 6.5, 1.0),  # Polar Hydrogen
        "B": FreeParams(46.7, 99.5, 2.08),
        "C": FreeParams(34.4, 46.6, 2.08),
        "N": FreeParams(25.9, 24.2, 1.72),
        "O": FreeParams(22.1, 15.6, 1.60),
        "F": FreeParams(18.2, 9.5, 1.58),
        "P": FreeParams(84.6, 185, 2.07),
        "S": FreeParams(75.2, 134.0, 2.00),
        "Cl": FreeParams(65.1, 94.6, 1.88),
        "Br": FreeParams(95.7, 162.0, 1.96),
        "Si": FreeParams(101.64, 305, 2.08),
        "I": FreeParams(153.8, 385.0, 2.04),
    }
    # If left as 1, 0, then no change will be made to final calc (multiply by 1 and to power of 0)
    alpha: float = 1
    beta: float = 0
    # with open('optimise.out') as opt_file:
    #     lines = opt_file.readlines()
    #     for i, line in enumerate(lines):
    #         if 'Final physical parameters:' in line:
    #             elem_dict['C'] = FreeParams(34.4, 46.6, float(lines[i + 2].split(' ')[6]))
    #             elem_dict['N'] = FreeParams(25.9, 24.2, float(lines[i + 3].split(' ')[6]))
    #             elem_dict['O'] = FreeParams(22.1, 15.6, float(lines[i + 4].split(' ')[6]))
    #             elem_dict['H'] = FreeParams(7.6, 6.5, float(lines[i + 5].split(' ')[6]))
    #             elem_dict['X'] = FreeParams(7.6, 6.5, float(lines[i + 6].split(' ')[6]))
    #             try:
    #                 alpha = float(lines[i + 7].split(' ')[2])
    #                 beta = float(lines[i + 8].split(' ')[2])
    #             except (IndexError, ValueError):
    #                 pass

    def __init__(self, molecule: Ligand):

        self.molecule: Ligand = molecule

        self.c8_params: Optional[List[float]] = None

        self.non_bonded_force: Dict[int: List[float, float, float]] = {}

    def extract_c8_params(self):
        """
        Extract the C8 dispersion coefficients from the MCLF calculation's output file.
        :return: c8_params ordered list of the c8 params for each atom in molecule
        """

        with open("MCLF_C8_dispersion_coefficients.xyz") as c8_file:
            lines = c8_file.readlines()
            for i, line in enumerate(lines):
                if line.startswith(" The following "):
                    lines = lines[i + 2 : -2]
                    break
            else:
                raise EOFError("Cannot locate c8 parameters in file.")

            # c8 params IN ATOMIC UNITS
            self.c8_params = [float(line.split()[-1].strip()) for line in lines]

    def append_ais_bis(self):
        """
        Use the AIM parameters from extract_params_*() to calculate a_i and b_i according to paper.
        Calculations from paper have been combined and simplified for faster computation.
        """

        for atom_index, atom in self.molecule.ddec_data.items():
            try:
                atomic_symbol, atom_vol = atom.atomic_symbol, atom.volume

                # Find polar Hydrogens and allocate their new name: X
                if atomic_symbol == "H":
                    bonded_index = self.molecule.atoms[atom_index].bonds[0]
                    if self.molecule.atoms[bonded_index].atomic_symbol in [
                        "N",
                        "O",
                        "S",
                    ]:
                        atomic_symbol = "X"

                # r_aim = r_free * ((vol / v_free) ** (1 / 3))
                r_aim = self.elem_dict[atomic_symbol].rfree * (
                    (atom_vol / self.elem_dict[atomic_symbol].vfree) ** (1 / 3)
                )

                # b_i = bfree * ((vol / v_free) ** 2)
                b_i = self.elem_dict[atomic_symbol].bfree * (
                    (atom_vol / self.elem_dict[atomic_symbol].vfree) ** 2
                )

                a_i = 32 * b_i * (r_aim ** 6)

            # Element not in elem_dict.
            except KeyError:
                r_aim, b_i, a_i = 0, 0, 0

            self.molecule.ddec_data[atom_index].r_aim = r_aim
            self.molecule.ddec_data[atom_index].b_i = b_i
            self.molecule.ddec_data[atom_index].a_i = a_i

    def calculate_sig_eps(self):
        """
        Adds the charge, sigma and epsilon terms to the ligand class object in a dictionary.
        The ligand class object (NonbondedForce) is stored as an empty dictionary until this method is called.
        """

        # Creates Nonbondedforce dict for later xml creation.
        # Format: {atom_index: [partial_charge, sigma, epsilon] ... }
        # This follows the usual ordering of the atoms such as in molecule.coords.
        for atom_index, atom in self.molecule.ddec_data.items():
            if not atom.a_i:
                sigma = epsilon = 0
            else:
                # sigma = (a_i / b_i) ** (1 / 6)
                sigma = (atom.a_i / atom.b_i) ** (1 / 6)
                sigma *= constants.SIGMA_CONVERSION

                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (atom.b_i ** 2) / (4 * atom.a_i)
                atomic_symbol = self.molecule.atoms[atom_index].atomic_symbol
                # alpha and beta
                epsilon *= self.alpha * (
                    (atom.volume / self.elem_dict[atomic_symbol].vfree) ** self.beta
                )
                epsilon *= constants.EPSILON_CONVERSION

            self.non_bonded_force[atom_index] = [atom.charge, sigma, epsilon]

    def correct_polar_hydrogens(self):
        """
        Identifies the polar Hydrogens and changes the a_i, b_i values accordingly.
        May be removed / heavily changed if we switch away from atom typing and use SMARTS.
        """

        # Loop through pairs in topology
        # Create new pair list with the atoms
        new_pairs = [
            (self.molecule.atoms[pair[0]], self.molecule.atoms[pair[1]])
            for pair in self.molecule.topology.edges
        ]

        # Find all the polar hydrogens and store their positions / atom numbers
        polars = []
        for pair in new_pairs:
            if (
                "O" == pair[0].atomic_symbol
                or "N" == pair[0].atomic_symbol
                or "S" == pair[0].atomic_symbol
            ):
                if "H" == pair[1].atomic_symbol:
                    polars.append(pair)

            if (
                "O" == pair[1].atomic_symbol
                or "N" == pair[1].atomic_symbol
                or "S" == pair[1].atomic_symbol
            ):
                if "H" == pair[0].atomic_symbol:
                    polars.append(pair)

        # Find square root of all b_i values so that they can be added easily according to paper's formula.
        for atom in self.molecule.ddec_data.values():
            atom.b_i = math.sqrt(atom.b_i)

        if polars:
            for pair in polars:
                if "H" == pair[0].atomic_symbol or "H" == pair[1].atomic_symbol:
                    if "H" == pair[0].atomic_symbol:
                        polar_h_pos = pair[0].atom_index
                        polar_son_pos = pair[1].atom_index
                    else:
                        polar_h_pos = pair[1].atom_index
                        polar_son_pos = pair[0].atom_index

                    # Calculate the new b_i for the two polar atoms (polar h and polar sulfur, oxygen or nitrogen)
                    self.molecule.ddec_data[
                        polar_son_pos
                    ].b_i += self.molecule.ddec_data[polar_h_pos].b_i
                    self.molecule.ddec_data[polar_h_pos].b_i = 0

        for atom in self.molecule.ddec_data.values():
            # Square all the b_i values again
            atom.b_i *= atom.b_i
            # Recalculate the a_is based on the new b_is
            atom.a_i = 32 * atom.b_i * (atom.r_aim ** 6)

        # Update epsilon (not sigma) according to new a_i and b_i values
        for atom_index, atom in self.molecule.ddec_data.items():
            if atom.a_i:
                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (atom.b_i ** 2) / (4 * atom.a_i)
                epsilon *= constants.EPSILON_CONVERSION
            else:
                epsilon, self.non_bonded_force[atom_index][1] = 0, 0

            self.non_bonded_force[atom_index] = [
                atom.charge,
                self.non_bonded_force[atom_index][1],
                epsilon,
            ]

    def calculate_non_bonded_force(self):
        """
        Main worker method for LennardJones class.
        Calculates the a_i and b_i values;
        Calculates the sigma and epsilon values using those a_i and b_i values;
        Redistributes L-J parameters according to polar Hydrogens, then recalculates epsilon values.
        """

        # Calculate initial a_is and b_is
        self.append_ais_bis()

        # Use the a_is and b_is to calculate the non_bonded_force dict
        self.calculate_sig_eps()

        # Tweak for polar Hydrogens
        # NB DISABLE FOR FORCEBALANCE
        self.correct_polar_hydrogens()

        self.molecule.NonbondedForce = self.non_bonded_force
