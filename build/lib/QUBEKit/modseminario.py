"""
Module to implement the Modified Seminario Method
Originally written by Alice E. A. Allen, TCM, University of Cambridge
Modified by Joshua T. Horton and rewritten by Chris Ringrose, Newcastle University
Reference using AEA Allen, MC Payne, DJ Cole, J. Chem. Theory Comput. (2018), doi:10.1021/acs.jctc.7b00785
"""

# TODO Josh finish getting the Seminario method to work from any qm engine hessian should be in np.array format

# Bugs
# TODO Fix repeated definitions (remove unused).

# Speed
# TODO Use context managers. (almost DONE, waiting on function fix.)
# TODO Convert while loops to for x in range().
#      (Each loop using while is 10 ops, only 3 using range, therefore ~3.3x faster. Also much more readable.)
# TODO Convert for item in range(len(items): to for item in items (where possible)
#      This also goes for read files, e.g. for line in file: rather than lines = file.readlines()
#                                                                        for line in lines:
#      There are currently 27 instances of readline(); there should be 0.
# TODO Pretty sure we can remove imports of os.path and itemgetter with simpler workarounds.

# Maintainability / Readability
# TODO Remove unused variables and functions. (almost DONE, waiting on function fix.)
# TODO Appropriately name variables (not just i, j, k etc).
# TODO Class structure.
#      Class with (mostly) static methods for dot product etc and class for main mod seminario work.
# TODO Fix formatting. (MOSTLY DONE)
# TODO Improve data structures. Dicts/sets/tuples over lists etc.
# TODO Change from list appending to generator expressions. (SOMEWHAT DONE)
# TODO Structure for searching files should be (for example):

# for line in file:
#     if line.startswith('Atomic numbers'):
#         >>>
#     if line.startswith('Current cartesian coordinates'):
#         >>>
#
# This will check all lines quickly and cleanly.


from QUBEKit.helpers import config_loader
from QUBEKit.decorators import for_all_methods, timer_logger

from numpy import cross, linalg, empty, zeros, array, reshape, dot, real
from numpy import append as npappend
from math import degrees, acos
from os.path import exists, isfile


class ModSemMaths:
    """Class largely will consist of static methods for various mathematical functions."""

    @staticmethod
    def unit_vector_n(u_bc, u_ab):
        """Calculates unit normal vector which is perpendicular to plane ABC."""

        return cross(u_bc, u_ab) / linalg.norm(cross(u_bc, u_ab))

    @staticmethod
    def u_pa_from_angles(atom_a, atom_b, atom_c, coords):
        """This gives the vector in the plane A, B, C and perpendicular to A to B."""

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
    def force_angle_constant(atom_a, atom_b, atom_c, bond_lengths, eigenvalues, eigenvectors, coords, scaling_1,
                             scaling_2):
        """Force Constant-Equation 14 of Seminario calculation paper-gives force
        constant for angle (in kcal/mol/rad^2) and equilibrium angle in degrees.
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
        sum_second = sum(
            eigenvalues_cb[i] * abs(ModSemMaths.dot_product(u_pc, eigenvectors_cb[:, i])) for i in range(3))

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
        confs = config_loader(config_dict['config'])
        self.qm, self.fitting, self.paths = confs

    def modified_seminario_method(self):
        """Calculate the new bond and angle terms after being passed the symmetric hessian and optimized
         molecule may also need the a parameter file."""

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

        # The final section finds the average bond and angle terms for each
        # bond/angle class if the .z exists to supply angle/bond classes and then
        # writes the new terms to a .sb file
        # if exists(f'{inputfilefolder}Zmat.z'):
        #     average_values_across_classes(unique_values_bonds, unique_values_angles)
        #     sb_file_new_parameters(outputfilefolder, 'Python_Modified_Scaled')

    def angles_calculated_printed(self, angle_list, bond_lengths, eigenvalues, eigenvectors, coords):
        """Uses the modified Seminario method to find the angle parameters and prints them to file"""

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
                    # For angle ABC, atoms A C are written to array
                    ac_array = [angle[0] - 1, angle[2] - 1, count]
                    central_atoms_angles[coord].append(ac_array)

                    # For angle ABC, atoms C A are written to array
                    ca_array = [angle[2] - 1, angle[0] - 1, count]
                    central_atoms_angles[coord].append(ca_array)

        # Sort rows by atom number
        for coord in range(len(coords)):
            central_atoms_angles[coord] = sorted(central_atoms_angles[coord], key=itemgetter(0))

        # Find normals u_PA for each angle
        unit_pa_all_angles = []

        for i in range(len(central_atoms_angles)):
            unit_pa_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                # For the angle at central_atoms_angles[i][j,:] the corresponding
                # u_PA value is found for the plane ABC and bond AB, where ABC
                # corresponds to the order of the arguments
                # This is why the reverse order was also added
                unit_pa_all_angles[i].append(ModSemMaths.u_pa_from_angles(central_atoms_angles[i][j][0], i, central_atoms_angles[i][j][1], coords))

        # Finds the contributing factors from the other angle terms
        # scaling_factor_all_angles = cell(max(max(angle_list))) This will contain scaling factor and angle list position
        scaling_factor_all_angles = []

        for i in range(len(central_atoms_angles)):
            scaling_factor_all_angles.append([])
            for j in range(len(central_atoms_angles[i])):
                n = m = 1
                angles_around = additional_contributions = 0
                scaling_factor_all_angles[i].append([0, 0])

                # Position in angle list
                scaling_factor_all_angles[i][j][1] = central_atoms_angles[i][j][2]

                # Goes through the list of angles with the same central atom
                # And computes the term need for the modified Seminario method

                # Forwards directions, finds the same bonds with the central atom i
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

        # Open output file angle parameters are written to
        with open('Modified_Seminario_Angle.txt', 'w') as angle_file:

        # Finds the angle force constants with the scaling factors included for each angle
            for i, angle in enumerate(angle_list):
                # Ensures that there is no difference when the ordering is changed
                [ab_k_theta, ab_theta_0] = ModSemMaths.force_angle_constant(angle[0] - 1, angle[1] - 1, angle[2] - 1, bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][0], scaling_factors_angles_list[i][1])
                [ba_k_theta, ba_theta_0] = ModSemMaths.force_angle_constant(angle[2] - 1, angle[1] - 1, angle[0] - 1, bond_lengths, eigenvalues, eigenvectors, coords, scaling_factors_angles_list[i][1], scaling_factors_angles_list[i][0])
                k_theta[i] = (ab_k_theta + ba_k_theta) / 2
                theta_0[i] = (ab_theta_0 + ba_theta_0) / 2

                # Vib_scaling takes into account DFT deficiencies / anharmonicity
                k_theta[i] *= (self.qm['vib scaling'] ** 2)



                angle_file.write(f'{i}  {self.atom_names[angle[0] - 1]}-{self.atom_names[angle[1] - 1]}-{self.atom_names[angle[2] - 1]}  ')

                angle_file.write('{:.3f}   {:.3f}   {}   {}   {}\n'.format(k_theta[i], theta_0[i], angle[0], angle[1], angle[2]))

                unique_values_angles.append([self.atom_names[angle[0] - 1], self.atom_names[angle[1] - 1], self.atom_names[angle[2] - 1], k_theta[i], theta_0[i], 1])

        return unique_values_angles

    def bonds_calculated_printed(self, bond_list, bond_lengths, eigenvalues, eigenvectors, coords):
        """Uses the Seminario method to find the bond parameters and print them to file"""

        # Open output file bond parameters are written to
        with open('Modified_Seminario_Bonds.txt', 'w+') as bond_file:
            k_b = zeros(len(bond_list))
            bond_length_list = zeros(len(bond_list))
            unique_values_bonds = []  # Used to find average values

            for i, bond in enumerate(bond_list):
                ab = ModSemMaths.force_constant_bond(bond[0] - 1, bond[1] - 1, eigenvalues, eigenvectors, coords)
                ba = ModSemMaths.force_constant_bond(bond[1] - 1, bond[0] - 1, eigenvalues, eigenvectors, coords)

                # Order of bonds sometimes causes slight differences, find the mean
                k_b[i] = real((ab + ba) / 2)

                # Vib_scaling takes into account DFT deficiencies/ anharmonicity
                k_b[i] *= (self.qm['vib scaling'] ** 2)

                bond_length_list[i] = bond_lengths[bond[0] - 1][bond[1] - 1]
                bond_file.write(f'{self.atom_names[bond[0] - 1]}-{self.atom_names[bond[1] - 1]}  ')
                bond_file.write('{:.3f}   {:.3f}   {}   {}\n'.format(k_b[i], bond_length_list[i], bond[0], bond[1]))

                unique_values_bonds.append(
                    [self.atom_names[bond[0] - 1], self.atom_names[bond[1] - 1], k_b[i], bond_length_list[i], 1])

        return unique_values_bonds


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


# TODO Broken (and the function that calls it is broken too).
def coords_from_fchk(fchk_file):
    """Function extracts xyz file from the .fchk output file from Gaussian.
    This provides the coordinates of the molecules."""

    if exists(fchk_file):
        fid = open(fchk_file, "r")
    else:
        raise FileNotFoundError('no .lig.fchk file found!')
        
    tline = fid.readline()

    numbers = []  # Atomic numbers for use in xyz file
    list_coords = []  # List of xyz coordinates
    hessian = []

    # Get atomic number and coordinates from fchk

    while tline:
        # Atomic Numbers found
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            while len(tline) < 17 or (tline[0:16].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()

        # Get coordinates
        if len(tline) > 31 and tline[0:31].strip() == 'Current cartesian coordinates':
            tline = fid.readline()
            while len(tline) < 15 or (tline[0:14].strip() != 'Force Field' and tline[0:17].strip() != 'Int Atom Types'and tline[0:13].strip() != 'Atom Types'):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()

        # Gets Hessian
        if len(tline) > 25 and (tline[0:24].strip() == 'Cartesian Force Constants'):
            tline = fid.readline()
            while len(tline) < 13 or (tline[0:12].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                npappend(hessian, tmp, 0)
                tline = fid.readline()  # TODO readline() called 7 times?!

    fid.close()

    list_coords = [float(x) * float(0.529) for x in list_coords]

    # Number of atoms
    N = len(list_coords) // 3

    # Opens the new xyz file
    file = open('input_coords.xyz', "w+")
    file.write(f'{N}\n\n')

    xyz = zeros((N, 3))
    n = 0

    # TODO this needs to be turned into a element dictionary if we are going to use g09
    fid_csv = open('elementlist.csv', "r")

    # TODO Broken.
    with fid_csv as f:
        lines = fid_csv.read().splitlines()

    # Turn list in a matrix, with elements containing atomic number, symbol and name
    element_names = [x.split(",") for x in lines]
        
    # Gives name for atomic number
    names = [element_names[int(x) - 1][1] for x in numbers]

    # Print coordinates to new input_coords.xyz file
    for i in range(N):
        for j in range(3):
            xyz[i][j] = list_coords[n]
            n += 1

        file.write(f'{names[i]}{round(xyz[i][0], 3)} {round(xyz[i][1], 3)} {round(xyz[i][2], 3)}\n')
    file.close()

    # TODO Bug returns the wrong n/N ? n is currently not used.
    return N


# TODO Function not called and broken; remove?
def input_data_processing_g09():
    """Extracts input coords and hessian from .fchk file;
    bond and angle lists from .log file; and atom names (if a z-matrix is supplied).
    """

    # Gets Hessian in unprocessed form and writes .xyz file too
    [unprocessed_Hessian, N, names, coords] = coords_from_fchk('lig.fchk')

    # Gets bond and angle lists
    # [bond_list, angle_list] = bond_angle_list_gaussian()
    #
    # with open("Number_to_Atom_type.txt") as f:
    #     OPLS_number_to_name = f.readlines()
    #
    # OPLS_number_to_name = [x.split() for x in OPLS_number_to_name]

    length_hessian = 3 * N
    hessian = zeros((length_hessian, length_hessian))
    m = 0

    # Write the hessian in a 2D array format
    for i in range(length_hessian):
        for j in range((i + 1)):
            hessian[i][j] = unprocessed_Hessian[m]
            hessian[j][i] = unprocessed_Hessian[m]
            m += 1

    # Change from Hartree/bohr to kcal/mol /ang
    # 2242.378318 = 627.509391 / (0.529 ** 2)
    hessian *= 2242.378318

    # if zmat exists part here 
    # atom_names = []
    #
    # for i in range(len(names)):
    #     atom_names.append(names[i].strip() + str(i + 1))
    #
    # if exists(f'{inputfilefolder}Zmat.z'):
    #     atom_names = []
    #
    #     fid = open(f'{inputfilefolder}Zmat.z') #Boss type Zmat
    #
    #     tline = fid.readline()
    #
    #     #Find number of dummy atoms
    #     number_dummy = 0
    #     tmp = tline.split()
    #
    #     while tmp[2] == '-1':
    #         number_dummy += 1
    #         tline = fid.readline()
    #         tmp = tline.split()
    #
    #     if int(tmp[3]) < 800:
    #         for i in range(N):
    #             for j in range(len(OPLS_number_to_name)):
    #                 if OPLS_number_to_name[j][0] == tmp[3]:
    #                     atom_names.append(OPLS_number_to_name[j][1])
    #
    #             tline = fid.readline()
    #             tmp = tline.split()
    #     else:
    #         #For CM1A format
    #         while len(tmp) < 2 or tmp[1] != 'Non-Bonded':
    #             tline = fid.readline()
    #             tmp = tline.split()
    #
    #         tline = fid.readline()
    #         tmp = tline.split()
    #
    #         for i in range(N):
    #             atom_names.append(tmp[2])
    #             tline = fid.readline()
    #             tmp = tline.split()
    #
    #     for i in range(N):
    #         if len(atom_names[i]) == 1:
    #             atom_names[i] += ' '
            
    # return(bond_list, angle_list, coords, N, hessian, atom_names)
    return hessian


# TODO Repeated function definition; remove?
def coords_from_fchk(fchk_file):
    """Extracts xyz file from the Guassian .fchk output file; this provides the coordinates of the molecules"""

    if exists(fchk_file):
        fid = open(fchk_file, "r")
    else:
        fid_log = open('MSM_log', "a")
        fid_log.write('ERROR = No .fchk file found.')
        fid_log.close()
        return 0, 0
        
    tline = fid.readline()

    numbers = []    # Atomic numbers for use in xyz file
    list_coords = []    # List of xyz coordinates
    hessian = []

    # Get atomic number and coordinates from fchk
    while tline:
        # Atomic Numbers found
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            while len(tline) < 17 or (tline[0:16].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()
            
        # Get coordinates
        if len(tline) > 31 and tline[0:31].strip() == 'Current cartesian coordinates':
            tline = fid.readline()
            while len(tline) < 15 or (tline[0:14].strip() != 'Force Field' and tline[0:17].strip() != 'Int Atom Types'and tline[0:13].strip() != 'Atom Types'):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()
            N = len(list_coords) // 3    # Number of atoms

        # Gets Hessian
        if len(tline) > 25 and (tline[0:26].strip() == 'Cartesian Force Constants'):
            tline = fid.readline()
            
            while len(tline) < 13 or (tline[0:14].strip() != 'Dipole Moment'):
                tmp = (tline.strip()).split()
                hessian.extend(tmp)
                tline = fid.readline()

        tline = fid.readline()

    fid.close()

    list_coords = [float(x) * float(0.529) for x in list_coords]

    # Opens the new xyz file
    file = open('input_coords.xyz', "w+")
    file.write(f'{N}\n\n')

    xyz = zeros((N, 3))
    n = 0

    fid_csv = open('elementlist.csv', "r")

    with fid_csv as f:
        lines = fid_csv.read().splitlines()

    # Turn list in a matrix, with elements containing atomic number, symbol and name
    element_names = [x.split(",") for x in lines]
        
    # Gives name for atomic number
    names = [element_names[int(x) - 1][1] for x in numbers]

    # Print coordinates to new input_coords.xyz file
    for i in range(N):
        for j in range(3):
            xyz[i][j] = list_coords[n]
            n += 1
        file.write(f'{names[i]}{round(xyz[i][0], 3)} {round(xyz[i][1], 3)} {round(xyz[i][2], 3)}\n')

    file.close()
    # TODO variable n not used; remove?
    return hessian, N, names, xyz


def bond_angle_list_gaussian():
    """Extracts a list of bond and angles from the Gaussian .log file"""

    fname = 'zmat.log'

    if isfile(fname):
        fid = open(fname, "r")
    elif isfile('/lig.log'):
        fid = open('/lig.log', "r")
    else:
        fid_log = open('MSM_log', "a")
        fid_log.write('ERROR - No .log file found. \n')
        return

    tline = fid.readline()
    bond_list = []
    angle_list = []

    tmp = 'R'   # States if bond or angle

    # Finds the bond and angles from the .log file
    while tline:
        tline = fid.readline()
        # Line starts at point when bond and angle list occurs
        if len(tline) > 80 and '! Name  Definition ' in tline[0:81].strip():
            tline = fid.readline()  # TODO readline() called 3 times?
            # Stops when all bond and angles recorded
            while (tmp[0] == 'R') or (tmp[0] == 'A'):
                line = tline.split()
                tmp = line[1]
                
                # Bond or angles listed as string
                list_terms = line[2][2:-1]

                # Bond List
                if tmp[0] == 'R':
                    x = list_terms.split(',')
                    x = [(int(i)) for i in x]
                    bond_list.append(x)

                # Angle List
                if tmp[0] == 'A':
                    x = list_terms.split(',')
                    x = [(int(i)) for i in x]
                    angle_list.append(x)

                tline = fid.readline()  # TODO 4 times?!

            # Leave loop
            tline = -1

    return bond_list, angle_list


def sb_file_new_parameters(inputfilefolder, filename):
    """Takes new angle and bond terms and puts them into a .sb file with name: filename_seminario.sb
    """

    with open(f'{inputfilefolder}Average_Modified_Seminario_Angles.txt', 'r') as angle_file, \
            open(f'{inputfilefolder}Average_Modified_Seminario_Bonds.txt', 'r') as bond_file:

        bonds = [line.strip().split('  ') for line in bond_file]
        angles = [line.strip().split('  ') for line in angle_file]

    # Script produces this file
    with open(f'{inputfilefolder}{filename}_Seminario.sb', 'wt') as fidout:

        fidout.write('*****                         Bond Stretching and Angle Bending Parameters - July 17*****\n')

        # Prints out bonds at top of file
        for bond in bonds:
            fidout.write(f'{bond[0]} {bond[1]}      {bond[2]}        Modified Seminario Method AEAA \n')

        fidout.write('\n********                        line above must be blank\n')

        # Prints out angles in middle of file
        for angle in angles:
            if len(angle) == 8:
                fidout.write(f'{angle[0]}    {angle[1]}       {angle[2]}    Modified Seminario Method AEAA \n\n\n')
            else:
                fidout.write(f'{angle[0]}     {angle[1]}       {angle[2]}        Modified Seminario Method AEAA \n\n\n')
