#!/usr/bin/env python3
import math
from typing import TYPE_CHECKING, Dict, Tuple

from pydantic import Field
from typing_extensions import Literal

from qubekit.nonbonded.utils import FreeParams, LJData
from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase
from qubekit.utils.exceptions import MissingRfreeError

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class LennardJones612(StageBase):
    type: Literal["LennardJones612"] = "LennardJones612"
    lj_on_polar_h: bool = Field(
        True,
        description="If polar hydrogen should keep their LJ values `True`, rather than transfer them to the parent atom `False`.",
    )
    free_parameters: Dict[str, FreeParams] = Field(
        ...,
        description="The Rfree parameters used to derive the Lennard Jones terms.",
    )
    # If left as 1, 0, then no change will be made to final calc (multiply by 1 and to power of 0)
    alpha: float = Field(
        default=1.0,
        description="The amount by which the aim/free volume ration should be scaled.",
    )
    beta: float = Field(
        default=0.0,
        description="The power by which the aim/free volume should raised. Note this will be 2 + beta.",
    )

    def start_message(self, **kwargs) -> str:
        return "Calculating Lennard-Jones parameters for a 12-6 potential."

    def finish_message(self, **kwargs) -> str:
        return "Lennard-Jones 12-6 parameters calculated."

    @classmethod
    def is_available(cls) -> bool:
        """This class should always be available."""
        return True

    def check_element_coverage(self, molecule: "Ligand"):
        """
        For the given molecule check that we have Rfree parameters for all of the present elements.

        Note:
            If polar hydrogens are to have LJ terms an Rfree must be given for element X
        """
        target_elements = set([atom.atomic_symbol.lower() for atom in molecule.atoms])
        covered_elements = set([e.lower() for e in self.free_parameters.keys()])
        missing_elements = target_elements.difference(covered_elements)
        if missing_elements != set():
            raise MissingRfreeError(
                "The following elements have no reference Rfree values which are required to "
                f"parameterise the molecule {missing_elements}"
            )

        if self.lj_on_polar_h and "x" not in covered_elements:
            raise MissingRfreeError(
                "Please supply Rfree data for polar hydrogen using the symbol X is `lj_on_polar_h` is True"
            )

    def _run(self, molecule: "Ligand", **kwargs) -> "Ligand":
        """
        Use the reference AIM data in the molecule to calculate the Non-bonded (non-electrostatic) terms for the forcefield.
        * Calculates the a_i, b_i and r_aim values.
        * Redistributes above values according to polar Hydrogens.
        * Calculates the sigma and epsilon values using those a_i and b_i values.
        * Stores the values in the molecule object.
        """

        self.check_element_coverage(molecule=molecule)

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
                    if (
                        molecule.atoms[bonded_index].atomic_symbol
                        in [
                            "N",
                            "O",
                            "S",
                        ]
                        and self.lj_on_polar_h
                    ):
                        atomic_symbol = "X"

                # r_aim = r_free * ((vol / v_free) ** (1 / 3))
                r_aim = self.free_parameters[atomic_symbol].r_free * (
                    (atom_vol / self.free_parameters[atomic_symbol].v_free) ** (1 / 3)
                )

                # b_i = bfree * ((vol / v_free) ** 2)
                b_i = (
                    self.free_parameters[atomic_symbol].b_free
                    * self.alpha
                    * (
                        (atom_vol / self.free_parameters[atomic_symbol].v_free)
                        ** (2 + self.beta)
                    )
                )

                a_i = 32 * b_i * (r_aim**6)

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
            lj_data[atom_index].a_i = 32 * lj_datum.b_i * (lj_datum.r_aim**6)

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

                # alpha and beta
                # epsilon *= self.alpha * (
                #     (atom.aim.volume / self.free_parameters[atom.atomic_symbol].v_free)
                #     ** self.beta
                # )
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
