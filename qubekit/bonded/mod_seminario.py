"""
Module to implement the Modified Seminario Method
Originally written by Alice E. A. Allen, TCM, University of Cambridge
Modified by Joshua T. Horton and rewritten by Chris Ringrose, Newcastle University
Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785
"""

import copy
from operator import itemgetter

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from qubekit.molecules import Ligand
from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase


class ModSemMaths:
    """Static methods for various mathematical functions relevant to the modified Seminario method."""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    @staticmethod
    def unit_vector_normal_to_bond(u_bc, u_ab):
        """Calculates unit vector which is normal to the plane abc."""

        cross = np.cross(u_bc, u_ab)

        return cross / np.linalg.norm(cross)

    @staticmethod
    def unit_vector_along_bond(coords, bond):
        """Calculates the unit vector along a bond."""

        atom_a, atom_b = bond
        diff_ab = coords[atom_b] - coords[atom_a]

        return diff_ab / np.linalg.norm(diff_ab)

    @staticmethod
    def u_pa_from_angles(angle, coords):
        """This gives the vector in the plane a, b, c and perpendicular to a to b."""

        atom_a, atom_b, atom_c = angle

        u_ab = ModSemMaths.unit_vector_along_bond(coords, (atom_a, atom_b))
        u_cb = ModSemMaths.unit_vector_along_bond(coords, (atom_c, atom_b))

        u_n = ModSemMaths.unit_vector_normal_to_bond(u_cb, u_ab)

        return ModSemMaths.unit_vector_normal_to_bond(u_n, u_ab)

    @staticmethod
    def dot_product(u_pa, eig_ab):

        return sum(u_pa[i] * eig_ab[i].conjugate() for i in range(3))

    @staticmethod
    def force_constant_bond(bond, eigenvals, eigenvecs, coords):
        """Force Constant - Equation 10 of Seminario paper - gives force constant for bond."""

        atom_a, atom_b = bond
        eigenvals_ab = eigenvals[atom_a, atom_b, :]
        eigenvecs_ab = eigenvecs[:, :, atom_a, atom_b]

        unit_vectors_ab = ModSemMaths.unit_vector_along_bond(coords, bond)

        return -0.5 * sum(
            eigenvals_ab[i] * abs(np.dot(unit_vectors_ab, eigenvecs_ab[:, i]))
            for i in range(3)
        )

    @staticmethod
    def force_constant_angle(angle, bond_lens, eigenvals, eigenvecs, coords, scalings):
        """
        Force Constant - Equation 14 of Seminario paper - gives force constant for angle
        (in kcal/mol/rad^2) and equilibrium angle (in degrees).
        """

        atom_a, atom_b, atom_c = angle

        u_ab = ModSemMaths.unit_vector_along_bond(coords, (atom_a, atom_b))
        u_cb = ModSemMaths.unit_vector_along_bond(coords, (atom_c, atom_b))

        bond_len_ab = bond_lens[atom_a, atom_b]
        eigenvals_ab = eigenvals[atom_a, atom_b, :]
        eigenvecs_ab = eigenvecs[:3, :3, atom_a, atom_b]

        bond_len_bc = bond_lens[atom_b, atom_c]
        eigenvals_cb = eigenvals[atom_c, atom_b, :]
        eigenvecs_cb = eigenvecs[:3, :3, atom_c, atom_b]

        # Normal vector to angle plane found
        u_n = ModSemMaths.unit_vector_normal_to_bond(u_cb, u_ab)

        # Angle is linear:
        if abs(np.linalg.norm(u_cb - u_ab)) < 0.01 or (
            1.99 < abs(np.linalg.norm(u_cb - u_ab)) < 2.01
        ):
            # Scalings are set to 1.
            k_theta, theta_0 = ModSemMaths.f_c_a_special_case(
                u_ab,
                u_cb,
                [bond_len_ab, bond_len_bc],
                [eigenvals_ab, eigenvals_cb],
                [eigenvecs_ab, eigenvecs_cb],
            )

        else:
            u_pa = ModSemMaths.unit_vector_normal_to_bond(u_n, u_ab)
            u_pc = ModSemMaths.unit_vector_normal_to_bond(u_cb, u_n)

            # Scaling due to additional angles - Modified Seminario Part
            sum_first = (
                sum(
                    eigenvals_ab[i]
                    * abs(ModSemMaths.dot_product(u_pa, eigenvecs_ab[:, i]))
                    for i in range(3)
                )
                / scalings[0]
            )
            sum_second = (
                sum(
                    eigenvals_cb[i]
                    * abs(ModSemMaths.dot_product(u_pc, eigenvecs_cb[:, i]))
                    for i in range(3)
                )
                / scalings[1]
            )

            # Added as two springs in series
            k_theta = (1 / ((bond_len_ab**2) * sum_first)) + (
                1 / ((bond_len_bc**2) * sum_second)
            )
            k_theta = 1 / k_theta

            # Change to OPLS form
            k_theta = abs(k_theta * 0.5)

            # Equilibrium Angle
            theta_0 = np.degrees(np.arccos(np.dot(u_ab, u_cb)))

        return k_theta, theta_0

    @staticmethod
    def f_c_a_special_case(u_ab, u_cb, bond_lens, eigenvals, eigenvecs):
        """
        Force constant angle special case, for example nitrile groups.
        This is for when the bond is linear, and therefore cannot be sampled around in the same way.
        The perpendicular vector is not defined for a linear bond.
        """

        # Number of samples around the bond.
        n_samples = 200
        k_theta_array = np.zeros(n_samples)

        for theta in range(n_samples):

            u_n = [
                np.sin(theta) * np.cos(theta),
                np.sin(theta) * np.sin(theta),
                np.cos(theta),
            ]

            u_pa = ModSemMaths.unit_vector_normal_to_bond(u_n, u_ab)
            u_pc = ModSemMaths.unit_vector_normal_to_bond(u_cb, u_n)

            sum_first = sum(
                eigenvals[0][i] * abs(ModSemMaths.dot_product(u_pa, eigenvecs[0][:, i]))
                for i in range(3)
            )
            sum_second = sum(
                eigenvals[1][i] * abs(ModSemMaths.dot_product(u_pc, eigenvecs[1][:, i]))
                for i in range(3)
            )

            k_theta_i = (1 / ((bond_lens[0] ** 2) * sum_first)) + (
                1 / ((bond_lens[1] ** 2) * sum_second)
            )
            k_theta_i = 1 / k_theta_i

            k_theta_array[theta] = abs(k_theta_i * 0.5)

        k_theta = np.average(k_theta_array)
        theta_0 = np.degrees(np.arccos(np.dot(u_ab, u_cb)))

        return k_theta, theta_0


class ModSeminario(StageBase):

    type: Literal["ModSeminario"] = "ModSeminario"
    vibrational_scaling: float = Field(
        1.0,
        description="The vibration scaling that should be used to correct the reference DFT frequencies.",
    )

    @classmethod
    def is_available(cls) -> bool:
        """This class is part of qubekit and always available."""
        return True

    def start_message(self, **kwargs) -> str:
        return "Calculating new bond and angle parameters with the modified Seminario method."

    def finish_message(self, **kwargs) -> str:
        return "Bond and angle parameters calculated."

    def run(self, molecule: Ligand, **kwargs) -> Ligand:
        """
        The main worker stage which takes the molecule and its hessian and calculates the modified seminario method.

        Args:
            molecule: The qubekit molecule class that should contain a valid hessian and optimised coordinates.

        Note:
            Please cite this method using <J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785>
        """

        # reset the bond and angle parameter groups
        molecule.BondForce.clear_parameters()
        molecule.AngleForce.clear_parameters()
        # convert the hessian from atomic units
        conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS**2)
        # make sure we do not change the molecule hessian
        hessian = copy.deepcopy(molecule.hessian)
        hessian *= conversion
        self._modified_seminario_method(molecule=molecule, hessian=hessian)
        # apply symmetry to the bond and angle parameters
        molecule.symmetrise_bonded_parameters()

        return molecule

    def _modified_seminario_method(
        self, molecule: Ligand, hessian: np.ndarray
    ) -> Ligand:
        """
        Calculate the new bond and angle terms after being passed the symmetric Hessian and
        optimised molecule coordinates.
        """
        size_mol = molecule.n_atoms
        eigenvecs = np.empty((3, 3, size_mol, size_mol), dtype=complex)
        eigenvals = np.empty((size_mol, size_mol, 3), dtype=complex)
        bond_lens = np.zeros((size_mol, size_mol))

        for i in range(size_mol):
            for j in range(size_mol):
                diff_i_j = molecule.coordinates[i, :] - molecule.coordinates[j, :]
                bond_lens[i, j] = np.linalg.norm(diff_i_j)

                partial_hessian = hessian[
                    (i * 3) : ((i + 1) * 3), (j * 3) : ((j + 1) * 3)
                ]

                eigenvals[i, j, :], eigenvecs[:, :, i, j] = np.linalg.eig(
                    partial_hessian
                )

        # The bond and angle values are calculated and written to file.
        self.calculate_bonds(eigenvals, eigenvecs, molecule, bond_lens)
        self.calculate_angles(eigenvals, eigenvecs, molecule, bond_lens)
        return molecule

    def calculate_angles(
        self, eigenvals, eigenvecs, molecule: Ligand, bond_lengths: np.ndarray
    ):
        """
        Uses the modified Seminario method to find the angle parameters and prints them to file.
        """

        # A structure is created with the index giving the central atom of the angle;
        # an array then lists the angles with that central atom.
        # e.g. central_atoms_angles[3] contains an array of angles with central atom 3.

        # Connectivity information for Modified Seminario Method
        central_atoms_angles = []

        for coord in range(molecule.n_atoms):
            central_atoms_angles.append([])
            for count, angle in enumerate(molecule.angles):
                if coord == angle[1]:
                    # For angle abc, atoms a, c are written to array
                    central_atoms_angles[coord].append([angle[0], angle[2], count])

                    # For angle abc, atoms c a are written to array
                    central_atoms_angles[coord].append([angle[2], angle[0], count])

        # Sort rows by atom number
        for coord in range(molecule.n_atoms):
            central_atoms_angles[coord] = sorted(
                central_atoms_angles[coord], key=itemgetter(0)
            )

        # Find normals u_pa for each angle
        unit_pa_all_angles = []

        for i in range(len(central_atoms_angles)):
            unit_pa_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                # For the angle at central_atoms_angles[i][j,:] the u_pa value is found for plane abc and bond ab,
                # where abc corresponds to the order of the arguments. This is why the reverse order was also added.
                angle = central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1]
                unit_pa_all_angles[i].append(
                    ModSemMaths.u_pa_from_angles(angle, molecule.coordinates)
                )

        # Finds the contributing factors from the other angle terms
        scaling_factor_all_angles = []

        for i in range(len(central_atoms_angles)):
            scaling_factor_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                n = m = 1
                angles_around = extra_contribs = 0
                scaling_factor_all_angles[i].append([0, 0])

                # Position in angle list
                scaling_factor_all_angles[i][j][1] = central_atoms_angles[i][j][2]

                # Goes through the list of angles with the same central atom, then computes the term needed for MSM.

                # Forwards direction, finds the same bonds with the central atom i
                while ((j + n) < len(central_atoms_angles[i])) and central_atoms_angles[
                    i
                ][j][0] == central_atoms_angles[i][j + n][0]:
                    extra_contribs += (
                        abs(
                            np.dot(
                                unit_pa_all_angles[i][j][:],
                                unit_pa_all_angles[i][j + n][:],
                            )
                        )
                    ) ** 2
                    n += 1
                    angles_around += 1

                # Backwards direction, finds the same bonds with the central atom i
                while ((j - m) >= 0) and central_atoms_angles[i][j][
                    0
                ] == central_atoms_angles[i][j - m][0]:
                    extra_contribs += (
                        abs(
                            np.dot(
                                unit_pa_all_angles[i][j][:],
                                unit_pa_all_angles[i][j - m][:],
                            )
                        )
                    ) ** 2
                    m += 1
                    angles_around += 1

                scaling_factor_all_angles[i][j][0] = 1
                if n != 1 or m != 1:
                    # Finds the mean value of the additional contribution
                    scaling_factor_all_angles[i][j][0] += extra_contribs / (m + n - 2)

        scaling_factors_angles_list = [[] for _ in range(molecule.n_angles)]

        # Orders the scaling factors according to the angle list
        for i in range(len(central_atoms_angles)):
            for j in range(len(central_atoms_angles[i])):
                scaling_factors_angles_list[scaling_factor_all_angles[i][j][1]].append(
                    scaling_factor_all_angles[i][j][0]
                )

        k_theta, theta_0 = np.zeros(len(molecule.angles)), np.zeros(
            len(molecule.angles)
        )

        conversion = constants.KCAL_TO_KJ * 2

        with open("Modified_Seminario_Angles.txt", "w") as angle_file:

            for i, angle in enumerate(molecule.angles):

                scalings = scaling_factors_angles_list[i]

                # Ensures that there is no difference when the ordering is changed.
                ab_k_theta, ab_theta_0 = ModSemMaths.force_constant_angle(
                    angle,
                    bond_lengths,
                    eigenvals,
                    eigenvecs,
                    molecule.coordinates,
                    scalings,
                )
                ba_k_theta, ba_theta_0 = ModSemMaths.force_constant_angle(
                    angle[::-1],
                    bond_lengths,
                    eigenvals,
                    eigenvecs,
                    molecule.coordinates,
                    scalings[::-1],
                )

                # Vib_scaling takes into account DFT deficiencies / anharmonicity.
                k_theta[i] = ((ab_k_theta + ba_k_theta) / 2) * (
                    self.vibrational_scaling**2
                )
                theta_0[i] = (ab_theta_0 + ba_theta_0) / 2

                angle_file.write(
                    f"{molecule.atoms[angle[0]].atom_name}-{molecule.atoms[angle[1]].atom_name}-{molecule.atoms[angle[2]].atom_name}  "
                )
                angle_file.write(
                    f"{k_theta[i]:.3f}   {theta_0[i]:.3f}   {angle[0]}   {angle[1]}   {angle[2]}\n"
                )

                # Add ModSem values to ligand object.
                molecule.AngleForce.create_parameter(
                    atoms=angle,
                    angle=theta_0[i] * constants.DEG_TO_RAD,
                    k=k_theta[i] * conversion,
                )

    def calculate_bonds(
        self, eigenvals, eigenvecs, molecule: Ligand, bond_lengths: np.ndarray
    ):
        """
        Uses the modified Seminario method to find the bond parameters and print them to file.
        """

        bonds = molecule.to_topology().edges
        conversion = constants.KCAL_TO_KJ * 200

        k_b, bond_len_list = np.zeros(len(bonds)), np.zeros(len(bonds))

        with open("Modified_Seminario_Bonds.txt", "w") as bond_file:

            for pos, bond in enumerate(bonds):
                ab = ModSemMaths.force_constant_bond(
                    bond, eigenvals, eigenvecs, molecule.coordinates
                )
                ba = ModSemMaths.force_constant_bond(
                    bond[::-1], eigenvals, eigenvecs, molecule.coordinates
                )

                # Order of bonds sometimes causes slight differences; find the mean and apply vib_scaling.
                k_b[pos] = np.real((ab + ba) / 2) * (self.vibrational_scaling**2)

                bond_len_list[pos] = bond_lengths[bond]
                bond_file.write(
                    f"{molecule.atoms[bond[0]].atom_name}-{molecule.atoms[bond[1]].atom_name}  "
                )
                bond_file.write(
                    f"{k_b[pos]:.3f}   {bond_len_list[pos]:.3f}   {bond[0]}   {bond[1]}\n"
                )

                # Add ModSem values to ligand object.
                molecule.BondForce.create_parameter(
                    atoms=bond, length=bond_len_list[pos] / 10, k=conversion * k_b[pos]
                )
