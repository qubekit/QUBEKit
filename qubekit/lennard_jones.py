#!/usr/bin/env python3

import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

from pydantic import dataclasses

from qubekit.molecules import Ligand
from qubekit.utils import constants


@dataclasses.dataclass
class LJData:
    a_i: float
    b_i: float
    r_aim: float


class LennardJones612:

    # Beware weird units, (wrong in the paper too).
    # Units: vfree: Bohr ** 3, bfree: Ha * (Bohr ** 6), rfree: Angs
    FreeParams = namedtuple("params", "vfree bfree rfree")
    elem_dict: Dict[str, FreeParams] = {
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

        self.non_bonded_force: Dict[int, List[float, float, float]] = {}

    def calculate_lj_data(self) -> Dict[int, LJData]:
        """
        Use the AIM parameters to calculate a_i and b_i according to paper.
        Calculations from paper have been combined and simplified for faster computation.
        returns: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
        """

        lj_data = {}

        for atom_index, atom in enumerate(self.molecule.atoms):
            try:
                atomic_symbol, atom_vol = atom.atomic_symbol, atom.aim.volume

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

            lj_data[atom_index] = LJData(a_i=a_i, b_i=b_i, r_aim=r_aim)
        return lj_data

    def correct_polar_hydrogens(self, lj_data: Dict[int, LJData]) -> Dict[int, LJData]:
        """
        Identifies the polar Hydrogens and changes the a_i, b_i values accordingly.
        May be removed / heavily changed if we switch away from atom typing and use SMARTS.
        Args:
            lj_data: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
        Returns:
            same dict, with the values altered to have their polar Hs corrected.
        """

        # Loop through pairs in topology
        # Create new pair list with the atoms
        new_pairs = [
            (self.molecule.atoms[pair[0]], self.molecule.atoms[pair[1]])
            for pair in self.molecule.to_topology().edges
        ]

        # Find all the polar hydrogens and store their positions / atom numbers
        polars = []
        # TODO Use smirks
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
        for atom_index, lj_datum in lj_data.items():
            lj_data[atom_index].b_i = math.sqrt(lj_datum.b_i)

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
                    lj_data[polar_son_pos].b_i += lj_data[polar_h_pos].b_i
                    lj_data[polar_h_pos].b_i = 0

        for atom_index, lj_datum in lj_data.items():
            # Square all the b_i values again
            lj_data[atom_index].b_i *= lj_datum.b_i
            # Recalculate the a_is based on the new b_is
            lj_data[atom_index].a_i = 32 * lj_datum.b_i * (lj_datum.r_aim ** 6)

        return lj_data

    def calculate_sig_eps(
        self, lj_data: Dict[int, LJData]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Use the lj_data to calculate the sigma and epsilon values
        Args:
            lj_data: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
        Returns:
            The calculated sigma and epsilon values ready to be inserted into the molecule object.
        """
        non_bonded_forces = {}

        for atom, lj_datum in zip(self.molecule.atoms, lj_data.values()):
            if not lj_datum.a_i:
                sigma, epsilon = 0, 0
            else:
                # sigma = (a_i / b_i) ** (1 / 6)
                sigma = (lj_datum.a_i / lj_datum.b_i) ** (1 / 6)
                sigma *= constants.SIGMA_CONVERSION

                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (lj_datum.b_i * lj_datum.b_i) / (4 * lj_datum.a_i)

                # alpha and beta
                epsilon *= self.alpha * (
                    (atom.aim.volume / self.elem_dict[atom.atomic_symbol].vfree)
                    ** self.beta
                )
                epsilon *= constants.EPSILON_CONVERSION

            non_bonded_forces[atom.atom_index] = (sigma, epsilon)

        return non_bonded_forces

    def calculate_non_bonded_force(self):
        """
        Main worker method for LennardJones class.
        * Calculates the a_i, b_i and r_aim values.
        * Redistributes above values according to polar Hydrogens.
        * Calculates the sigma and epsilon values using those a_i and b_i values.
        * Stores the values in the molecule object.
        """

        # Calculate initial a_is and b_is
        lj_data = self.calculate_lj_data()

        # Tweak for polar Hydrogens
        # NB DISABLE FOR FORCEBALANCE
        lj_data = self.correct_polar_hydrogens(lj_data)

        # Use the a_is and b_is to calculate the non_bonded_force dict
        non_bonded_forces = self.calculate_sig_eps(lj_data)

        # update the Nonbonded force using api
        for atom_index, (sigma, epsilon) in non_bonded_forces.items():
            nonbond_data = {
                "charge": self.molecule.atoms[atom_index].aim.charge,
                "sigma": sigma,
                "epsilon": epsilon,
            }
            self.molecule.NonbondedForce.create_parameter(
                atoms=(atom_index,), **nonbond_data
            )
