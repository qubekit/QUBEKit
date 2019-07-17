#!/usr/bin/env python3

"""
Module to implement the Modified Seminario Method
Originally written by Alice E. A. Allen, TCM, University of Cambridge
Modified by Joshua T. Horton and rewritten by Chris Ringrose, Newcastle University
Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785
"""

from QUBEKit.utils import constants
from QUBEKit.utils.decorators import for_all_methods, timer_logger

from operator import itemgetter

import numpy as np


class ModSemMaths:
    """Static methods for various mathematical functions relevant to the modified Seminario method."""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    @staticmethod
    def unit_vector_n(u_bc, u_ab):
        """Calculates unit normal vector which is perpendicular to plane abc."""

        return np.cross(u_bc, u_ab) / np.linalg.norm(np.cross(u_bc, u_ab))

    @staticmethod
    def vector_along_bond(coords, bond):

        diff_ab = coords[bond[1], :] - coords[bond[0], :]

        return diff_ab / np.linalg.norm(diff_ab)

    @staticmethod
    def u_pa_from_angles(angle, coords):
        """This gives the vector in the plane a, b, c and perpendicular to a to b."""

        atom_a, atom_b, atom_c = angle

        u_ab = ModSemMaths.vector_along_bond(coords, (atom_a, atom_b))
        u_cb = ModSemMaths.vector_along_bond(coords, (atom_c, atom_b))

        u_n = ModSemMaths.unit_vector_n(u_cb, u_ab)

        return ModSemMaths.unit_vector_n(u_n, u_ab)

    @staticmethod
    def dot_product(u_pa, eig_ab):

        return sum(u_pa[i] * eig_ab[i].conjugate() for i in range(3))

    @staticmethod
    def force_constant_bond(bond, eigenvals, eigenvecs, coords):
        """Force Constant - Equation 10 of Seminario paper - gives force constant for bond."""

        eigenvals_ab = eigenvals[bond[0], bond[1], :]
        eigenvecs_ab = eigenvecs[:, :, bond[0], bond[1]]

        unit_vectors_ab = ModSemMaths.vector_along_bond(coords, bond)

        return -0.5 * sum(eigenvals_ab[i] * abs(np.dot(unit_vectors_ab, eigenvecs_ab[:, i])) for i in range(3))

    @staticmethod
    def force_constant_angle(angle, bond_lens, eigenvals, eigenvecs, coords, scalings):
        """
        Force Constant - Equation 14 of Seminario paper - gives force constant for angle
        (in kcal/mol/rad^2) and equilibrium angle (in degrees).
        """

        atom_a, atom_b, atom_c = angle

        u_ab = ModSemMaths.vector_along_bond(coords, (atom_a, atom_b))
        u_cb = ModSemMaths.vector_along_bond(coords, (atom_c, atom_b))

        bond_len_ab = bond_lens[atom_a, atom_b]
        eigenvals_ab = eigenvals[atom_a, atom_b, :]
        eigenvecs_ab = eigenvecs[:3, :3, atom_a, atom_b]

        bond_len_bc = bond_lens[atom_b, atom_c]
        eigenvals_cb = eigenvals[atom_c, atom_b, :]
        eigenvecs_cb = eigenvecs[:3, :3, atom_c, atom_b]

        # Normal vector to angle plane found
        u_n = ModSemMaths.unit_vector_n(u_cb, u_ab)

        if abs(sum(u_cb - u_ab)) < 0.01 or (1.99 < abs(sum(u_cb - u_ab)) < 2.01):
            # Scalings are set to 1.
            k_theta, theta_0 = ModSemMaths.f_c_a_special_case(
                u_ab, u_cb, [bond_len_ab, bond_len_bc], [eigenvals_ab, eigenvals_cb], [eigenvecs_ab, eigenvecs_cb])

        else:
            u_pa = ModSemMaths.unit_vector_n(u_n, u_ab)
            u_pc = ModSemMaths.unit_vector_n(u_cb, u_n)

            # Scaling due to additional angles - Modified Seminario Part
            sum_first = sum(eigenvals_ab[i] * abs(ModSemMaths.dot_product(u_pa, eigenvecs_ab[:, i])) for i in range(3)) / scalings[0]
            sum_second = sum(eigenvals_cb[i] * abs(ModSemMaths.dot_product(u_pc, eigenvecs_cb[:, i])) for i in range(3)) / scalings[1]

            # Added as two springs in series
            k_theta = (1 / ((bond_len_ab ** 2) * sum_first)) + (1 / ((bond_len_bc ** 2) * sum_second))
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

            u_n = [np.sin(theta) * np.cos(theta), np.sin(theta) * np.sin(theta), np.cos(theta)]

            u_pa = ModSemMaths.unit_vector_n(u_n, u_ab)
            u_pc = ModSemMaths.unit_vector_n(u_cb, u_n)

            sum_first = sum(eigenvals[0][i] * abs(ModSemMaths.dot_product(u_pa, eigenvecs[0][:, i])) for i in range(3))
            sum_second = sum(eigenvals[1][i] * abs(ModSemMaths.dot_product(u_pc, eigenvecs[1][:, i])) for i in range(3))

            k_theta_i = (1 / ((bond_lens[0] ** 2) * sum_first)) + (1 / ((bond_lens[1] ** 2) * sum_second))
            k_theta_i = 1 / k_theta_i

            k_theta_array[theta] = abs(k_theta_i * 0.5)

        k_theta = np.average(k_theta_array)
        theta_0 = np.degrees(np.arccos(np.dot(u_ab, u_cb)))

        return k_theta, theta_0


@for_all_methods(timer_logger)
class ModSeminario:

    def __init__(self, molecule):

        self.molecule = molecule
        self.size_mol = len(self.molecule.atoms)
        # Find bond lengths and create empty matrix of correct size.
        self.bond_lens = np.zeros((self.size_mol, self.size_mol))
        self.coords = self.molecule.coords['qm']

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def modified_seminario_method(self):
        """
        Calculate the new bond and angle terms after being passed the symmetric Hessian and
        optimised molecule coordinates.
        """

        eigenvecs = np.empty((3, 3, self.size_mol, self.size_mol), dtype=complex)
        eigenvals = np.empty((self.size_mol, self.size_mol, 3), dtype=complex)

        for i in range(self.size_mol):
            for j in range(self.size_mol):
                diff_i_j = self.coords[i, :] - self.coords[j, :]
                self.bond_lens[i, j] = np.linalg.norm(diff_i_j)

                partial_hessian = self.molecule.hessian[(i * 3):((i + 1) * 3), (j * 3):((j + 1) * 3)]

                eigenvals[i, j, :], eigenvecs[:, :, i, j] = np.linalg.eig(partial_hessian)

        # The bond and angle values are calculated and written to file.
        self.calculate_bonds(eigenvals, eigenvecs)
        self.calculate_angles(eigenvals, eigenvecs)

    def calculate_angles(self, eigenvals, eigenvecs):
        """Uses the modified Seminario method to find the angle parameters and prints them to file."""

        # A structure is created with the index giving the central atom of the angle;
        # an array then lists the angles with that central atom.
        # e.g. central_atoms_angles[3] contains an array of angles with central atom 3.

        # Connectivity information for Modified Seminario Method
        central_atoms_angles = []

        for coord in range(self.size_mol):
            central_atoms_angles.append([])
            for count, angle in enumerate(self.molecule.angles):
                if coord == angle[1]:
                    # For angle abc, atoms a, c are written to array
                    central_atoms_angles[coord].append([angle[0], angle[2], count])

                    # For angle abc, atoms c a are written to array
                    central_atoms_angles[coord].append([angle[2], angle[0], count])

        # Sort rows by atom number
        for coord in range(self.size_mol):
            central_atoms_angles[coord] = sorted(central_atoms_angles[coord], key=itemgetter(0))

        # Find normals u_pa for each angle
        unit_pa_all_angles = []

        for i in range(len(central_atoms_angles)):
            unit_pa_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                # For the angle at central_atoms_angles[i][j,:] the u_pa value is found for plane abc and bond ab,
                # where abc corresponds to the order of the arguments. This is why the reverse order was also added.
                angle = central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1]
                unit_pa_all_angles[i].append(ModSemMaths.u_pa_from_angles(angle, self.coords))

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
                while ((j + n) < len(central_atoms_angles[i])) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j + n][0]:
                    extra_contribs += (abs(np.dot(unit_pa_all_angles[i][j][:], unit_pa_all_angles[i][j + n][:]))) ** 2
                    n += 1
                    angles_around += 1

                # Backwards direction, finds the same bonds with the central atom i
                while ((j - m) >= 0) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j - m][0]:
                    extra_contribs += (abs(np.dot(unit_pa_all_angles[i][j][:], unit_pa_all_angles[i][j - m][:]))) ** 2
                    m += 1
                    angles_around += 1

                scaling_factor_all_angles[i][j][0] = 1
                if n != 1 or m != 1:
                    # Finds the mean value of the additional contribution
                    scaling_factor_all_angles[i][j][0] += (extra_contribs / (m + n - 2))

        scaling_factors_angles_list = [[]] * len(self.molecule.angles)

        # Orders the scaling factors according to the angle list
        for i in range(len(central_atoms_angles)):
            for j in range(len(central_atoms_angles[i])):
                scaling_factors_angles_list[scaling_factor_all_angles[i][j][1]].append(scaling_factor_all_angles[i][j][0])

        # Used to find average values
        unique_values_angles = []
        k_theta, theta_0 = np.zeros(len(self.molecule.angles)), np.zeros(len(self.molecule.angles))

        conversion = constants.KCAL_TO_KJ * 2

        with open('Modified_Seminario_Angles.txt', f'{"w" if self.molecule.restart else "a+"}') as angle_file:

            for i, angle in enumerate(self.molecule.angles):

                scalings = scaling_factors_angles_list[i][:2]

                # Ensures that there is no difference when the ordering is changed.
                ab_k_theta, ab_theta_0 = ModSemMaths.force_constant_angle(angle, self.bond_lens, eigenvals, eigenvecs, self.coords, scalings)
                ba_k_theta, ba_theta_0 = ModSemMaths.force_constant_angle(angle[::-1], self.bond_lens, eigenvals, eigenvecs, self.coords, scalings[::-1])

                # Vib_scaling takes into account DFT deficiencies / anharmonicity.
                k_theta[i] = (self.molecule.vib_scaling ** 2) * ((ab_k_theta + ba_k_theta) / 2)
                theta_0[i] = (ab_theta_0 + ba_theta_0) / 2

                angle_file.write(f'{self.molecule.atoms[angle[0]].atom_name}-{self.molecule.atoms[angle[1]].atom_name}-{self.molecule.atoms[angle[2]].atom_name}  ')
                angle_file.write(f'{k_theta[i]:.3f}   {theta_0[i]:.3f}   {angle[0]}   {angle[1]}   {angle[2]}\n')

                # Add ModSem values to ligand object.
                self.molecule.HarmonicAngleForce[angle] = [theta_0[i] * np.pi / 180, k_theta[i] * conversion]

                unique_values_angles.append(
                    [self.molecule.atoms[angle[0]].atom_name, self.molecule.atoms[angle[1]].atom_name, self.molecule.atoms[angle[2]].atom_name,
                     k_theta[i] * conversion, theta_0[i] * np.pi / 180, 1])

        return unique_values_angles

    def calculate_bonds(self, eigenvals, eigenvecs):
        """Uses the modified Seminario method to find the bond parameters and print them to file."""

        bonds = self.molecule.topology.edges
        conversion = constants.KCAL_TO_KJ * 200

        k_b, bond_len_list = np.zeros(len(bonds)), np.zeros(len(bonds))

        # Used to find average values
        unique_values_bonds = []

        with open('Modified_Seminario_Bonds.txt', f'{"w" if self.molecule.restart else "a+"}') as bond_file:

            for pos, bond in enumerate(bonds):
                ab = ModSemMaths.force_constant_bond(bond, eigenvals, eigenvecs, self.coords)
                ba = ModSemMaths.force_constant_bond(bond[::-1], eigenvals, eigenvecs, self.coords)

                # Order of bonds sometimes causes slight differences; find the mean and apply vib_scaling.
                k_b[pos] = np.real((ab + ba) / 2) * (self.molecule.vib_scaling ** 2)

                bond_len_list[pos] = self.bond_lens[bond]
                bond_file.write(f'{self.molecule.atoms[bond[0]].atom_name}-{self.molecule.atoms[bond[1]].atom_name}  ')
                bond_file.write(f'{k_b[pos]:.3f}   {bond_len_list[pos]:.3f}   {bond[0]}   {bond[1]}\n')

                # Add ModSem values to ligand object.
                self.molecule.HarmonicBondForce[bond] = [bond_len_list[pos] / 10, conversion * k_b[pos]]

                unique_values_bonds.append(
                    [self.molecule.atoms[bond[0]].atom_name, self.molecule.atoms[bond[1]].atom_name,
                     k_b[pos] * conversion, bond_len_list[pos] / 10, 1])

        return unique_values_bonds
