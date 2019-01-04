"""
Module to implement the Modified Seminario Method
Originally written by Alice E. A. Allen, TCM, University of Cambridge
Modified by Joshua T. Horton and rewritten by Chris Ringrose, Newcastle University
Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785
"""

# TODO Convert 'while x:' loops to 'for x in range():' or 'for x in y:'.
#      (Each loop using while is 10 ops, only 3 using range, therefore ~3.3x faster. Also much more readable.)
# TODO Convert for item in range(len(items): to for item in items (where possible)

# Maintainability / Readability
# TODO Move hanging functions. (Almost DONE, average_values_across_classes requires rewrite.)
# TODO Appropriately name variables (not just i, j, k etc).
# TODO Improve data structures. Dicts/sets/tuples over lists etc.
# TODO Change from list appending to generator expressions. (SOMEWHAT DONE)


from QUBEKit.helpers import config_loader
from QUBEKit.decorators import for_all_methods, timer_logger

from numpy import cross, linalg, empty, zeros, array, reshape, dot, real
from math import degrees, acos


class ModSemMaths:
    """Static methods for various mathematical functions relevant to the modified Seminario method."""

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    @staticmethod
    def unit_vector_n(u_bc, u_ab):
        """Calculates unit normal vector which is perpendicular to plane abc."""

        return cross(u_bc, u_ab) / linalg.norm(cross(u_bc, u_ab))

    @staticmethod
    def u_pa_from_angles(atom_a, atom_b, atom_c, coords):
        """This gives the vector in the plane a, b, c and perpendicular to a to b."""

        diff_ab = coords[atom_b, :] - coords[atom_a, :]
        u_ab = diff_ab / linalg.norm(diff_ab)

        diff_cb = coords[atom_b, :] - coords[atom_c, :]
        u_cb = diff_cb / linalg.norm(diff_cb)

        u_n = ModSemMaths.unit_vector_n(u_cb, u_ab)

        u_pa = cross(u_n, u_ab) / linalg.norm(cross(u_n, u_ab))

        return u_pa

    @staticmethod
    def dot_product(u_pa, eig_ab):

        return sum(u_pa[i] * eig_ab[i].conjugate() for i in range(3))

    @staticmethod
    def force_constant_bond(atom_a, atom_b, eigenvalues, eigenvectors, coords):
        """Force Constant - Equation 10 of Seminario paper - gives force constant for bond."""

        # Eigenvalues and vectors calculated
        eigenvalues_ab = eigenvalues[atom_a, atom_b, :]
        eigenvectors_ab = eigenvectors[:, :, atom_a, atom_b]

        # Vector along bond
        diff_ab = array(coords[atom_b, :]) - array(coords[atom_a, :])

        unit_vectors_ab = diff_ab / linalg.norm(diff_ab)

        return -0.5 * sum(eigenvalues_ab[i] * abs(dot(unit_vectors_ab, eigenvectors_ab[:, i])) for i in range(3))

    @staticmethod
    def force_angle_constant(atom_a, atom_b, atom_c, bond_lengths, eigenvalues, eigenvectors, coords, scaling_1, scaling_2):
        """Force Constant - Equation 14 of Seminario paper - gives force constant for angle
        (in kcal/mol/rad^2) and equilibrium angle (in degrees).
        """

        # Vectors along bonds calculated
        diff_ab = coords[atom_b, :] - coords[atom_a, :]
        u_ab = diff_ab / linalg.norm(diff_ab)

        diff_cb = coords[atom_b, :] - coords[atom_c, :]
        u_cb = diff_cb / linalg.norm(diff_cb)

        # Bond lengths and eigenvalues found
        bond_length_ab = bond_lengths[atom_a, atom_b]
        eigenvalues_ab = eigenvalues[atom_a, atom_b, :]
        eigenvectors_ab = eigenvectors[0:3, 0:3, atom_a, atom_b]

        bond_length_bc = bond_lengths[atom_b, atom_c]
        eigenvalues_cb = eigenvalues[atom_c, atom_b, :]
        eigenvectors_cb = eigenvectors[0:3, 0:3, atom_c, atom_b]

        # Normal vector to angle plane found
        u_n = ModSemMaths.unit_vector_n(u_cb, u_ab)

        u_pa = cross(u_n, u_ab) / linalg.norm(cross(u_n, u_ab))
        u_pc = cross(u_cb, u_n) / linalg.norm(cross(u_cb, u_n))

        sum_first = sum(eigenvalues_ab[i] * abs(ModSemMaths.dot_product(u_pa, eigenvectors_ab[:, i])) for i in range(3))
        sum_second = sum(eigenvalues_cb[i] * abs(ModSemMaths.dot_product(u_pc, eigenvectors_cb[:, i])) for i in range(3))

        # Scaling due to additional angles - Modified Seminario Part
        sum_first /= scaling_1
        sum_second /= scaling_2

        # Added as two springs in series
        k_theta = (1 / ((bond_length_ab ** 2) * sum_first)) + (1 / ((bond_length_bc ** 2) * sum_second))
        k_theta = 1 / k_theta

        # Change to OPLS form
        k_theta = abs(-k_theta * 0.5)

        # Equilibrium Angle
        theta_0 = degrees(acos(dot(u_ab, u_cb)))

        return k_theta, theta_0


@for_all_methods(timer_logger)
class ModSeminario:

    def __init__(self, mol, config_dict):

        self.molecule = mol
        self.atom_names = self.molecule.atom_names
        # Load the configs using the config_file name.
        self.qm, self.fitting, self.descriptions = config_loader(config_dict['config'])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def modified_seminario_method(self):
        """Calculate the new bond and angle terms after being passed the symmetric Hessian and optimised
        molecule coordinates.
        """

        # Take required parameters from the molecule object; always use QMoptimized structure
        hessian = self.molecule.hessian
        size_mol = len(self.molecule.QMoptimized)
        bond_list = self.molecule.topology.edges  # with pdb numbering starting from 1 not 0
        angle_list = self.molecule.angles

        coords = [atom[j + 1] for atom in self.molecule.QMoptimized for j in range(3)]
        coords = reshape(coords, (size_mol, 3))

        # Find bond lengths and create empty matrix of correct size.
        bond_lengths = zeros((size_mol, size_mol))

        eigenvectors = empty((3, 3, size_mol, size_mol), dtype=complex)
        eigenvalues = empty((size_mol, size_mol, 3), dtype=complex)

        for i in range(size_mol):
            for j in range(size_mol):

                diff_i_j = array(coords[i, :]) - array(coords[j, :])
                bond_lengths[i][j] = linalg.norm(diff_i_j)

                partial_hessian = hessian[(i * 3):((i + 1) * 3), (j * 3):((j + 1) * 3)]
                [a, b] = linalg.eig(partial_hessian)

                eigenvalues[i, j, :] = a
                eigenvectors[:, :, i, j] = b

        # The bond and angle values are calculated and written to file.
        self.bonds_calculated_printed(bond_list, bond_lengths, eigenvalues, eigenvectors, coords)
        self.angles_calculated_printed(angle_list, bond_lengths, eigenvalues, eigenvectors, coords)

        # TODO Move/remove.
        # The final section finds the average bond and angle terms for each
        # bond/angle class if the .z exists to supply angle/bond classes and then
        # writes the new terms to a .sb file
        # if exists(f'{inputfilefolder}Zmat.z'):
        #     average_values_across_classes(unique_values_bonds, unique_values_angles)
        #     sb_file_new_parameters(outputfilefolder, 'Python_Modified_Scaled')

    def angles_calculated_printed(self, angle_list, bond_lengths, eigenvalues, eigenvectors, coords):
        """Uses the modified Seminario method to find the angle parameters and prints them to file."""

        from operator import itemgetter

        k_theta = theta_0 = zeros(len(angle_list))

        # Connectivity information for Modified Seminario Method
        central_atoms_angles = []

        # A structure is created with the index giving the central atom of the angle;
        # an array then lists the angles with that central atom.
        # e.g. central_atoms_angles{3} contains an array of angles with central atom 3.

        for coord in range(len(coords)):
            central_atoms_angles.append([])
            for count, angle in enumerate(angle_list):
                if coord == angle[1] - 1:
                    # For angle abc, atoms a, c are written to array
                    ac_array = [angle[0] - 1, angle[2] - 1, count]
                    central_atoms_angles[coord].append(ac_array)

                    # For angle abc, atoms c a are written to array
                    ca_array = [angle[2] - 1, angle[0] - 1, count]
                    central_atoms_angles[coord].append(ca_array)

        # Sort rows by atom number
        for coord in range(len(coords)):
            central_atoms_angles[coord] = sorted(central_atoms_angles[coord], key=itemgetter(0))

        # Find normals u_pa for each angle
        unit_pa_all_angles = []

        for i in range(len(central_atoms_angles)):
            unit_pa_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                # For the angle at central_atoms_angles[i][j,:] the u_pa value is found for plane abc and bond ab,
                # where abc corresponds to the order of the arguments. This is why the reverse order was also added.
                unit_pa_all_angles[i].append(ModSemMaths.u_pa_from_angles(central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1], coords))

        # Finds the contributing factors from the other angle terms
        scaling_factor_all_angles = []

        for i in range(len(central_atoms_angles)):
            scaling_factor_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                n = m = 1
                angles_around = additional_contributions = 0
                scaling_factor_all_angles[i].append([0, 0])

                # Position in angle list
                scaling_factor_all_angles[i][j][1] = central_atoms_angles[i][j][2]

                # Goes through the list of angles with the same central atom, then computes the term needed for modSem.

                # Forwards direction, finds the same bonds with the central atom i
                while ((j + n) < len(central_atoms_angles[i])) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j + n][0]:
                    additional_contributions += (abs(dot(unit_pa_all_angles[i][j][:], unit_pa_all_angles[i][j + n][:]))) ** 2
                    n += 1
                    angles_around += 1

                # Backwards direction, finds the same bonds with the central atom i
                while ((j - m) >= 0) and central_atoms_angles[i][j][0] == central_atoms_angles[i][j - m][0]:
                    additional_contributions += (abs(dot(unit_pa_all_angles[i][j][:], unit_pa_all_angles[i][j - m][:]))) ** 2
                    m += 1
                    angles_around += 1

                scaling_factor_all_angles[i][j][0] = 1
                if n != 1 or m != 1:
                    # Finds the mean value of the additional contribution
                    scaling_factor_all_angles[i][j][0] += (additional_contributions / (m + n - 2))

        scaling_factors_angles_list = [[]] * len(angle_list)

        # Orders the scaling factors according to the angle list
        for i in range(len(central_atoms_angles)):
            for j in range(len(central_atoms_angles[i])):
                scaling_factors_angles_list[scaling_factor_all_angles[i][j][1]].append(scaling_factor_all_angles[i][j][0])

        # Used to find average values
        unique_values_angles = []

        with open('Modified_Seminario_Angle.txt', 'w+') as angle_file:

            for i, angle in enumerate(angle_list):
                # Ensures that there is no difference when the ordering is changed.
                ab_k_theta, ab_theta_0 = ModSemMaths.force_angle_constant(angle[0] - 1, angle[1] - 1, angle[2] - 1, bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][0], scaling_factors_angles_list[i][1])
                ba_k_theta, ba_theta_0 = ModSemMaths.force_angle_constant(angle[2] - 1, angle[1] - 1, angle[0] - 1, bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][1], scaling_factors_angles_list[i][0])
                k_theta[i] = (ab_k_theta + ba_k_theta) / 2
                theta_0[i] = (ab_theta_0 + ba_theta_0) / 2

                # Vib_scaling takes into account DFT deficiencies / anharmonicity.
                k_theta[i] *= (self.qm['vib scaling'] ** 2)

                angle_file.write(f'{i}  {self.atom_names[angle[0] - 1]}-{self.atom_names[angle[1] - 1]}-{self.atom_names[angle[2] - 1]}  ')
                angle_file.write('{:.3f}   {:.3f}   {}   {}   {}\n'.format(k_theta[i], theta_0[i], angle[0], angle[1], angle[2]))

                unique_values_angles.append([self.atom_names[angle[0] - 1], self.atom_names[angle[1] - 1], self.atom_names[angle[2] - 1], k_theta[i], theta_0[i], 1])

        return unique_values_angles

    def bonds_calculated_printed(self, bond_list, bond_lengths, eigenvalues, eigenvectors, coords):
        """Uses the modified Seminario method to find the bond parameters and print them to file."""

        conversion = 418.4

        with open('Modified_Seminario_Bonds.txt', 'w+') as bond_file:

            k_b = zeros(len(bond_list))
            bond_length_list = zeros(len(bond_list))

            # Used to find average values
            unique_values_bonds = []

            for i, bond in enumerate(bond_list):
                ab = ModSemMaths.force_constant_bond(bond[0] - 1, bond[1] - 1, eigenvalues, eigenvectors, coords)
                ba = ModSemMaths.force_constant_bond(bond[1] - 1, bond[0] - 1, eigenvalues, eigenvectors, coords)

                # Order of bonds sometimes causes slight differences; find the mean.
                k_b[i] = real((ab + ba) / 2)

                # Vib_scaling takes into account DFT deficiencies/ anharmonicity.
                k_b[i] *= (self.qm['vib scaling'] ** 2)

                bond_length_list[i] = bond_lengths[bond[0] - 1][bond[1] - 1]
                bond_file.write(f'{self.atom_names[bond[0] - 1]}-{self.atom_names[bond[1] - 1]}  ')
                bond_file.write('{:.3f}   {:.3f}   {}   {}\n'.format(k_b[i], bond_length_list[i], bond[0], bond[1]))

                # Add ModSem values to ligand object.
                self.molecule.HarmonicBondForce[(bond[0] - 1, bond[1] - 1)] = [str(bond_length_list[i] / 10), str(conversion * k_b[i])]

                unique_values_bonds.append([self.atom_names[bond[0] - 1], self.atom_names[bond[1] - 1], k_b[i], bond_length_list[i], 1])

        return unique_values_bonds


# TODO This whole function should be scrapped and rewritten.
def average_values_across_classes(unique_values_bonds, unique_values_angles):
    """Finds the average bond and angle term for each class."""

    ignore_rows_bonds = []

    # Find Average Values Bonds
    for i in range(len(unique_values_bonds)):
        for j in range(i + 1, len(unique_values_bonds)):
            # Finds if the bond class has already been encountered
            if (unique_values_bonds[i][0] == unique_values_bonds[j][0]) and (unique_values_bonds[i][1] == unique_values_bonds[j][1]) or ((unique_values_bonds[i][0] == unique_values_bonds[j][1]) and (unique_values_bonds[i][1] == unique_values_bonds[j][0])):
                unique_values_bonds[i][2] += unique_values_bonds[j][2]
                unique_values_bonds[i][3] += unique_values_bonds[j][3]
                unique_values_bonds[i][4] += 1
                ignore_rows_bonds.append(j)

    # Average Bonds Printed
    with open('Average_Modified_Seminario_Bonds.txt', 'w+') as bond_file:

        # Remove bond classes that were already present and find mean value
        for i in range(len(unique_values_bonds)):
            if i not in ignore_rows_bonds:
                unique_values_bonds[i][2] /= unique_values_bonds[i][4]
                unique_values_bonds[i][3] /= unique_values_bonds[i][4]
                bond_file.write('{}-{}  {:.2f}  {:.3f}\n'.format(unique_values_bonds[i][0], unique_values_bonds[i][1], unique_values_bonds[i][2], unique_values_bonds[i][3]))
    # Find average values angles
    ignore_rows_angles = []

    # Find Average Values Angles
    for i in range(len(unique_values_angles)):
        for j in range(i + 1, len(unique_values_angles)):
            # Finds if the angle class has already been encountered
            if (unique_values_angles[i][0] == unique_values_angles[j][0] and unique_values_angles[i][1] == unique_values_angles[j][1] and unique_values_angles[i][2] == unique_values_angles[j][2]) or (unique_values_angles[i][0] == unique_values_angles[j][2] and unique_values_angles[i][1] == unique_values_angles[j][1] and unique_values_angles[i][2] == unique_values_angles[j][0]):
                unique_values_angles[i][3] += unique_values_angles[j][3]
                unique_values_angles[i][4] += unique_values_angles[j][4]
                unique_values_angles[i][5] += 1
                ignore_rows_angles.append(j)

    # Average Angles Printed
    with open('Average_Modified_Seminario_Angles.txt', 'w+') as angle_file:

        # Remove angles classes that were already present and find mean value
        for i in range(len(unique_values_angles)):
            if i not in ignore_rows_angles:
                unique_values_angles[i][3] /= unique_values_angles[i][5]
                unique_values_angles[i][4] /= unique_values_angles[i][5]
                angle_file.write('{}-{}-{}  {:.2f}  {:.3f}\n'.format(unique_values_angles[i][0], unique_values_angles[i][1], unique_values_angles[i][2], unique_values_angles[i][3], unique_values_angles[i][4]))


# TODO Move to different file (probably ligand file).
def sb_file_new_parameters(inputfilefolder, filename):
    """Takes new angle and bond terms and puts them into an sb file with name: filename_seminario.sb"""

    with open(f'{inputfilefolder}Average_Modified_Seminario_Angles.txt', 'r') as angle_file, \
            open(f'{inputfilefolder}Average_Modified_Seminario_Bonds.txt', 'r') as bond_file:

        angles = [line.strip().split('  ') for line in angle_file]
        bonds = [line.strip().split('  ') for line in bond_file]

    # Script produces this file
    with open(f'{inputfilefolder}{filename}_Seminario.sb', 'wt') as fidout:

        # TODO Use properly formatted spacing / padding; add headers for each column.
        fidout.write('*****                         Bond Stretching and Angle Bending Parameters - July 17*****\n')

        # Prints out bonds at top of file
        for bond in bonds:
            fidout.write(f'{bond[0]} {bond[1]}      {bond[2]}        Modified Seminario Method AEAA \n')

        fidout.write('\n********                        line above must be blank\n')

        # Prints out angles in middle of file
        for angle in angles:
            fidout.write(f'{angle[0]}    {angle[1]}       {angle[2]}    Modified Seminario Method AEAA \n\n\n')
