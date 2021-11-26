#!/usr/bin/env python3
import math
import os
from typing import TYPE_CHECKING, ClassVar, Dict, Tuple

from pydantic import Field, PrivateAttr
from typing_extensions import Literal

from qubekit.nonbonded.utils import FreeParams, LJData
from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class LennardJones612(StageBase):

    type: Literal["LennardJones612"] = "LennardJones612"
    lj_on_polar_h: bool = Field(
        False,
        description="If polar hydrogen should keep their LJ values `True`, rather than transfer them to the parent atom `False`.",
    )
    free_parameters: ClassVar[Dict[str, FreeParams]] = {
        # v_free, b_free, r_free
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
    _alpha: float = PrivateAttr(default=1)
    _beta: float = PrivateAttr(default=0)

    def start_message(self, **kwargs) -> str:
        return "Calculating Lennard-Jones parameters for a 12-6 potential."

    def finish_message(self, **kwargs) -> str:
        return "Lennard-Jones 12-6 parameters calculated."

    @classmethod
    def is_available(cls) -> bool:
        """This class should always be available."""
        return True

    def extract_rfrees(self):
        """
        Open any .out files from ForceBalance and read in the relevant parameters.
        Parameters are taken from the "Final physical parameters" section and stored
        to be used in the proceeding LJ calculations.
        """
        for file in os.listdir("../../"):
            if file.endswith(".out"):
                with open(f"../../{file}") as opt_file:
                    lines = opt_file.readlines()
                    for i, line in enumerate(lines):
                        if "Final physical parameters" in line:
                            lines = lines[i:]
                            break
                    else:
                        raise EOFError(
                            "Could not find final parameters in ForceBalance file."
                        )
                for line in lines:
                    for k, v in self.free_parameters.items():
                        if f"{k}Element" in line:
                            self.free_parameters[k] = FreeParams(
                                v.v_free, v.b_free, float(line.split(" ")[6])
                            )
                            print(f"Updated {k}free parameter from ForceBalance file.")
                    if "xalpha" in line:
                        self._alpha = float(line.split(" ")[6])
                        print("Updated alpha parameter from ForceBalance file.")
                    elif "xbeta" in line:
                        self._beta = float(line.split(" ")[6])
                        print("Updated beta parameter from ForceBalance file.")

    def run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Use the reference AIM data in the molecule to calculate the Non-bonded (non-electrostatic) terms for the forcefield.
        * Calculates the a_i, b_i and r_aim values.
        * Redistributes above values according to polar Hydrogens.
        * Calculates the sigma and epsilon values using those a_i and b_i values.
        * Stores the values in the molecule object.
        """

        self.extract_rfrees()

        # Calculate initial a_is and b_is
        lj_data = self._calculate_lj_data(molecule=molecule)

        # Tweak for polar Hydrogens
        # NB DISABLE FOR FORCEBALANCE
        if not self.lj_on_polar_h:
            lj_data = LennardJones612._correct_polar_hydrogens(
                lj_data, molecule=molecule
            )

        # Use the a_is and b_is to calculate the non_bonded_force dict
        non_bonded_forces = self._calculate_sig_eps(lj_data, molecule=molecule)

        # update the Nonbonded force using api
        for atom_index, (sigma, epsilon) in non_bonded_forces.items():
            nonbond_data = {
                "sigma": sigma,
                "epsilon": epsilon,
            }
            parameter = molecule.NonbondedForce[(atom_index,)]
            # update only the nonbonded parts in place
            parameter.update(**nonbond_data)

        return molecule

    def _calculate_lj_data(self, molecule: "Ligand") -> Dict[int, LJData]:
        """
        Use the AIM parameters to calculate a_i and b_i according to paper.
        Calculations from paper have been combined and simplified for faster computation.
        returns: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
        """

        lj_data = {}

        for atom_index, atom in enumerate(molecule.atoms):
            try:
                atomic_symbol, atom_vol = atom.atomic_symbol, atom.aim.volume

                # Find polar Hydrogens and allocate their new name: X
                if atomic_symbol == "H":
                    bonded_index = atom.bonds[0]
                    if molecule.atoms[bonded_index].atomic_symbol in [
                        "N",
                        "O",
                        "S",
                    ]:
                        atomic_symbol = "X"

                # r_aim = r_free * ((vol / v_free) ** (1 / 3))
                r_aim = self.free_parameters[atomic_symbol].r_free * (
                    (atom_vol / self.free_parameters[atomic_symbol].v_free) ** (1 / 3)
                )

                # b_i = bfree * ((vol / v_free) ** 2)
                b_i = self.free_parameters[atomic_symbol].b_free * (
                    (atom_vol / self.free_parameters[atomic_symbol].v_free) ** 2
                )

                a_i = 32 * b_i * (r_aim ** 6)

            # Element not in elem_dict.
            except KeyError:
                r_aim, b_i, a_i = 0, 0, 0

            lj_data[atom_index] = LJData(a_i=a_i, b_i=b_i, r_aim=r_aim)
        return lj_data

    @staticmethod
    def _correct_polar_hydrogens(
        lj_data: Dict[int, LJData], molecule: "Ligand"
    ) -> Dict[int, LJData]:
        """
        Identifies the polar Hydrogens and changes the a_i, b_i values accordingly.
        May be removed / heavily changed if we switch away from atom typing and use SMARTS.
        Args:
            lj_data: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
            molecule: The molecule that should be used to determine polar bonds.
        Returns:
            same dict, with the values altered to have their polar Hs corrected.
        """

        # Loop through pairs in topology
        # Create new pair list with the atoms
        new_pairs = [
            (molecule.atoms[bond.atom1_index], molecule.atoms[bond.atom2_index])
            for bond in molecule.bonds
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

    def _calculate_sig_eps(
        self,
        lj_data: Dict[int, LJData],
        molecule: "Ligand",
    ) -> Dict[int, Tuple[float, float]]:
        """
        Use the lj_data to calculate the sigma and epsilon values
        Args:
            lj_data: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
            molecule: The molecule we should calculate the non-bonded values for.
        Returns:
            The calculated sigma and epsilon values ready to be inserted into the molecule object.
        """
        non_bonded_forces = {}

        for atom, lj_datum in zip(molecule.atoms, lj_data.values()):
            if not lj_datum.a_i:
                sigma, epsilon = 1, 0
            else:
                # sigma = (a_i / b_i) ** (1 / 6)
                sigma = (lj_datum.a_i / lj_datum.b_i) ** (1 / 6)
                sigma *= constants.SIGMA_CONVERSION

                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (lj_datum.b_i * lj_datum.b_i) / (4 * lj_datum.a_i)

                # _alpha and _beta
                epsilon *= self._alpha * (
                    (atom.aim.volume / self.free_parameters[atom.atomic_symbol].v_free)
                    ** self._beta
                )
                epsilon *= constants.EPSILON_CONVERSION

            non_bonded_forces[atom.atom_index] = (sigma, epsilon)

        return non_bonded_forces

    def _calculate_b_prime(
        self,
        lj_data: Dict[int, LJData],
        molecule: "Ligand",
    ) -> Dict[int, Tuple[float, float]]:
        """
        Use the lj_data to calculate the sigma values, as well as a b_free_prime value.
        b_free_prime is used purely for FB optimisation, not normal MD simulations.
        ordinarily, epsilon = b_free_prime / (128 * r_free ** 6)
        However, to use in an FB optimisation, we must input b_free_prime instead of epsilon directly,
        since r_free needs to be optimised.
        Args:
            lj_data: Dict of the a_i, b_i and r_aim values needed for sigma/epsilon calculation.
            molecule: The molecule we should calculate the non-bonded values for.
        Returns:
            The calculated sigma and b_free_prime (called epsilon as placeholder) values
            ready to be inserted into the molecule object.
        """

        non_bonded_forces = {}

        for atom, lj_datum in zip(molecule.atoms, lj_data.values()):
            atomic_symbol, atom_vol = atom.atomic_symbol, atom.aim.volume
            if not lj_datum.a_i:
                sigma, epsilon = 1, 0
            else:
                sigma = (lj_datum.a_i / lj_datum.b_i) ** (1 / 6)
                sigma *= constants.SIGMA_CONVERSION

                # Used purely for FB optimisation.
                # eps = b_free_prime / (128 * r_free ** 6)
                b_free_prime = lj_datum.b_i / (
                    (atom_vol / self.free_parameters[atomic_symbol].v_free) ** 2
                )

                # rename for inserting into dict.
                epsilon = b_free_prime

            non_bonded_forces[atom.atom_index] = (sigma, epsilon)

        return non_bonded_forces
