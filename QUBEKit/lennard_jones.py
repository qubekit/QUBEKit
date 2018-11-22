#!/usr/bin/env python


# TODO Symmetry checks.
# TODO Check if Hydrogens are polar.


class LennardJones:

    def __init__(self, molecule, ddec_version=6):

        # Ligand class object
        self.molecule = molecule
        self.ddec_version = ddec_version

    def extract_params(self):
        """Extract the useful information from the DDEC xyz files.
        Prepare this information for the Lennard-Jones coefficient calculations."""

        # Get number of atoms from start of ddec file.

        # Extract atom types and numbers
        # Extract charges
        # Extract dipoles
        # Extract volumes (from other file)

        # Ensure total charge ~= net charge

        # return info for the molecule as a list of lists.

        if self.ddec_version == 6:
            net_charge_file_name = 'DDEC6_even_tempered_net_atomic_charges.xyz'

        elif self.ddec_version == 3:
            net_charge_file_name = 'DDEC3_net_atomic_charges.xyz'

        else:
            raise ValueError('Invalid or unsupported DDEC version.')

        molecule_data = []

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

                        molecule_data.append(atom_data)
                    break

        r_cubed_file_name = 'DDEC_atomic_Rcubed_moments.xyz'

        with open(r_cubed_file_name, 'r+') as vol_file:

            lines = vol_file.readlines()

            vols = []

            for line in lines[2:atom_total + 2]:

                vol = float(line.split()[-1])
                vols.append(vol)

            for count, atom in enumerate(molecule_data):
                atom.append(vols[count])

            # Ensure total charge is near to integer value:
            total_charge = 0
            for atom in molecule_data:
                total_charge += atom[5]

            # If not 0 < total_charge << 1: you've a problem.
            if round(total_charge) - total_charge > 0.00001:
                raise ValueError('Total charge is not close enough to integer value.')

        return molecule_data

    def calc_ai_bi(self):
        """Use the atom in molecule parameters from lj_extract_params to calculate the coefficients
        of the Lennard Jones Potential.
        Exact calculations are described in full and truncated form in the comments of the function.
        """

        # Calculate sigma and epsilon according to paper calcs
        # Calculations from paper have been combined and simplified for faster computation.

        mol = self.extract_params()

        # 'elem' : [vfree, bfree, rfree]

        # TODO Test values need to be removed or changed later (Zn, Au)
        elem_dict = {
            'H': [7.6, 6.5, 1.64],
            'C': [34.4, 46.6, 2.08],
            'N': [25.9, 24.2, 1.72],
            'O': [22.1, 15.6, 1.60],
            'F': [18.2, 9.5, 1.58],
            'S': [75.2, 134.0, 2.00],
            'Cl': [65.1, 94.6, 1.88],
            'Br': [95.7, 162.0, 1.96],
            'Zn': [1, 1, 1],
            'Au': [1, 1, 1]
        }

        # Conversion from Ha.Bohr ** 6 to kcal / (mol * Ang ** 6):
        kcal_ang = 13.7792544

        bis = []
        ais = []

        for atom in mol:

            # r_aim = rfree * ((vol / vfree) ** (1 / 3))
            r_aim = elem_dict[f'{atom[1]}'][2] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** (1 / 3))

            # b_i = bfree * ((vol / vfree) ** 2)
            b_i = elem_dict[f'{atom[1]}'][1] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** 2)

            a_i = 32 * b_i * (r_aim ** 6)

            bis.append(b_i)
            ais.append(a_i)

        return bis, ais

    def polar_hydrogens(self):
        """Identifies the polar Hydrogens and changes the sigma, epsilon values accordingly."""

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

        bis, ais = self.calc_ai_bi()

        for pair in polars:
            if 'H' in pair[0]:
                # pair[0] is the first atom of the tuple with its topology position e.g. '6H'
                # pair[0][0] - 1 is the index of the atom e.g. 5
                bis[int(pair[0][0]) - 1] = 0
                ais[int(pair[0][0]) - 1] = 0
            if 'H' in pair[1]:
                bis[int(pair[1][0]) - 1] = 0
                ais[int(pair[1][0]) - 1] = 0

        # Redistribute sigma/epsilon values to the polar neighbour
        # eq 10 in paper.
        # set the sigma/epsilon values to 0 for the found polar hydrogens

        return bis, ais

    def symmetry(self):
        pass

    def amend_ai_bi(self):

        self.molecule.NonbondedForce = {self.molecule.molecule[atom][0]: self.calc_ai_bi()[atom] for atom in range(len(self.molecule.molecule))}
