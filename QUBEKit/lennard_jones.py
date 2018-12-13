#!/usr/bin/env python


# TODO Symmetry checks.
# TODO Check if Hydrogens are polar.


from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.helpers import config_loader


@for_all_methods(timer_logger)
class LennardJones:

    def __init__(self, molecule, config_dict):

        # Ligand class object
        self.molecule = molecule
        self.qm, self.fitting, self.descriptions = config_loader(config_dict['config'])
        # This is the DDEC molecule data in the format:
        # ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x dipole', 'y dipole', 'z dipole', vol]
        self.ddec_data = self.extract_params()
        self.ddec_ai_bi = self.append_ais_bis()
        self.ddec_polars = self.polar_hydrogens()

    def extract_params(self):
        """Extract the useful information from the DDEC xyz files.
        Prepare this information for the Lennard-Jones coefficient calculations."""

        # Get number of atoms from start of ddec file.

        # Extract atom types and numbers
        # Extract charges
        # Extract dipoles
        # Extract volumes (from other file)

        # Ensure total charge ~== net charge

        # return info for the molecule as a list of lists.

        if self.qm['ddec version'] == 6:
            net_charge_file_name = 'DDEC6_even_tempered_net_atomic_charges.xyz'

        elif self.qm['ddec version'] == 3:
            net_charge_file_name = 'DDEC3_net_atomic_charges.xyz'

        else:
            raise ValueError('Invalid or unsupported DDEC version.')

        self.ddec_data = []

        with open(net_charge_file_name, 'r+') as charge_file:

            lines = charge_file.readlines()

            # Find number of atoms
            atom_total = int(lines[0])

            for count, row in enumerate(lines):

                if 'The following XYZ' in row:

                    start_pos = count + 2

                    for line in lines[start_pos:start_pos + atom_total]:
                        # Append the atom number and type, coords, charge, dipoles:
                        # ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x dipole', 'y dipole', 'z dipole']
                        atom_string_list = line.split()
                        # Append all the float values first.
                        atom_data = atom_string_list[2:9]
                        atom_data = [float(datum) for datum in atom_data]

                        # Prepend the first two values (atom_type = str, atom_number = int)
                        atom_data.insert(0, atom_string_list[1])
                        atom_data.insert(0, int(atom_string_list[0]))

                        self.ddec_data.append(atom_data)
                    break

        r_cubed_file_name = 'DDEC_atomic_Rcubed_moments.xyz'

        with open(r_cubed_file_name, 'r+') as vol_file:

            lines = vol_file.readlines()

            vols = [float(line.split()[-1]) for line in lines[2:atom_total + 2]]

            for count, atom in enumerate(self.ddec_data):
                atom.append(vols[count])

            # Ensure total charge is near to integer value:
            total_charge = 0
            for atom in self.ddec_data:
                total_charge += atom[5]

            # If not 0 < total_charge << 1: you've a problem.
            if abs(round(total_charge) - total_charge) > 0.00001:
                raise ValueError('Total charge is not close enough to integer value.')

        return self.ddec_data

    def append_ais_bis(self):
        """Use the atom in molecule parameters from extract_params to calculate the coefficients
        of the Lennard-Jones Potential.
        """

        # Calculate a_i and b_i according to paper calcs
        # Calculations from paper have been combined and simplified for faster computation.

        # 'elem' : [vfree, bfree, rfree]
        # TODO Remove Au and Zn after testing.
        elem_dict = {
            'H': [7.6, 6.5, 1.64],
            'C': [34.4, 46.6, 2.08],
            'N': [25.9, 24.2, 1.72],
            'O': [22.1, 15.6, 1.60],
            'F': [18.2, 9.5, 1.58],
            'S': [75.2, 134.0, 2.00],
            'Cl': [65.1, 94.6, 1.88],
            'Br': [95.7, 162.0, 1.96],
            'Au': [1, 1, 1],
            'Zn': [1, 1, 1]
        }

        for count, atom in enumerate(self.ddec_data):

            # r_aim = rfree * ((vol / vfree) ** (1 / 3))
            r_aim = elem_dict[f'{atom[1]}'][2] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** (1 / 3))

            # b_i = bfree * ((vol / vfree) ** 2)
            b_i = elem_dict[f'{atom[1]}'][1] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** 2)

            a_i = 32 * b_i * (r_aim ** 6)
            self.ddec_data[count] += [r_aim, b_i, a_i]

        return self.ddec_data

    def polar_hydrogens(self):
        """Identifies the polar Hydrogens and changes the a_i, b_i values accordingly."""

        # Create dictionary which stores the atom number and its type:
        # atoms = {1: 'C', 2: 'C', 3: 'H', 4: 'H', ...}
        # (+1 because topology indices count from 1, not 0)
        positions = {self.molecule.molecule.index(atom) + 1: atom[0] for atom in self.molecule.molecule}

        # Loop through pairs in topology
        # Create new pair list with atom types and positions using the dictionary:
        # new_pairs = [('1C', '3H'), ('1C', '4H'), ('1C', '5H') ...]
        new_pairs = []
        for pair in self.molecule.topology.edges:
            new_pair = (str(pair[0]) + positions[pair[0]], str(pair[1]) + positions[pair[1]])
            new_pairs.append(new_pair)

        # Find all the polar hydrogens and store their positions/atom numbers
        polars = []
        for pair in new_pairs:
            if 'O' in pair[0] or 'N' in pair[0] or 'S' in pair[0]:
                if 'H' in pair[1]:
                    polars.append(pair)
            if 'O' in pair[1] or 'N' in pair[1] or 'S' in pair[1]:
                if 'H' in pair[0]:
                    polars.append(pair)

        print('Polar pairs identified: ', polars)

        for pair in polars:
            if 'H' in pair[0] or 'H' in pair[1]:
                if 'H' in pair[0]:
                    polar_h_pos = int(pair[0][0]) - 1
                    polar_son_pos = int(pair[1][0]) - 1
                else:
                    polar_h_pos = int(pair[1][0]) - 1
                    polar_son_pos = int(pair[0][0]) - 1
                # Reset the b_i for the two polar atoms (polar h and polar sulfur, oxygen or nitrogen)
                self.ddec_ai_bi[polar_son_pos][-2] = ((self.ddec_ai_bi[polar_son_pos][-2]) ** 0.5 + (self.ddec_ai_bi[polar_h_pos][-2]) ** 0.5) ** 2
                self.ddec_ai_bi[polar_h_pos][-2] = 0

                # Reset the a_i for the two polar atoms using the new b_i values.
                self.ddec_ai_bi[polar_son_pos][-1] = 32 * self.ddec_ai_bi[polar_son_pos][-2] * (self.ddec_ai_bi[polar_son_pos][-3] ** 6)
                self.ddec_ai_bi[polar_h_pos][-1] = 0

        return self.ddec_ai_bi

    def symmetry(self):
        """Symmetrises the sigma and epsilon terms.
        This means setting the sigma and epsilon values to be equal if the atom types are the same.
        """

        # Symmetrisation may be carried out in different ways.

        # 1. Work backwards
        # Perform a charges calculation for each atom.
        # If the charge on a set of atoms is largely consistent then set their terms to be equal to their average.
        # Use these values to then calculate the L-J terms which are now symmetrised.

        # 2. Use the molecular structure
        # Examine the topology of the molecule and find similar atoms.
        # Look at the nearest neighbour atoms, as well as the nearest neighbours of the nearest neighbours.
        # This should be enough to symmetrise the atoms in MOST cases.
        # It is not sufficient to simply look at the nearest neighbours and no further.
        # This leads to "over-symmetrisation" where different atom types are labelled as the same.

        pass

    def amend_sig_eps(self):
        """Adds the sigma, epsilon terms to the ligand class object as a dictionary.
        The class object (NonbondedForce) is stored as an empty dictionary until this method is called.
        # TODO Add the sigma/epsilon terms after symmetry fixes.
        """

        # Creates Nonbondedforce dict:
        # Format: {0: [sigma, epsilon], 1: [sigma, epsilon], ... }
        # This follows the usual ordering of the atoms which is consistent throughout QUBEKit.

        new_NonbondedForce = {}
        # Conversion from Ha.Bohr ** 6 to kcal / (mol * Ang ** 6):
        kcal_ang = 13.7792544

        for atom in range(len(self.molecule.molecule)):
            if self.ddec_polars[atom][-1] == 0:
                sigma, epsilon = 0, 0
            else:
                sigma = (self.ddec_polars[atom][-1] / self.ddec_polars[atom][-2]) ** (1 / 6)
                epsilon = (self.ddec_polars[atom][-2] ** 2) / (4 * self.ddec_polars[atom][-1])

            new_NonbondedForce.update({atom: [self.ddec_polars[atom][5], sigma, epsilon]})

        self.molecule.NonbondedForce = new_NonbondedForce

        return self.molecule
