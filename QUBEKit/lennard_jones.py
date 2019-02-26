#!/usr/bin/env python


# TODO Symmetry checks, revisit polar hydrogens; ensure correct atoms are being adjusted.


from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.helpers import check_net_charge

from os.path import exists


@for_all_methods(timer_logger)
class LennardJones:

    def __init__(self, molecule, config_dict):

        # Ligand class object
        self.molecule = molecule
        self.defaults_dict, self.qm, self.fitting, self.descriptions = config_dict
        # self.ddec_data is the DDEC molecule data in the format:
        # ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x dipole', 'y dipole', 'z dipole', 'vol']
        # It will be extended and tweaked by each core method of this class.
        # self.amend_sig_eps() can then be called after class initialisation which will return the
        # nonBondedForce dict for the xml writer.
        self.ddec_data = []

        if self.qm['charges_engine'] == 'chargemol':
            self.extract_params_chargemol()

        elif self.qm['charges_engine'] == 'onetep':
            self.extract_params_onetep()

        else:
            raise KeyError('Invalid Charges engine provided, cannot extract charges.')

        # Calculate initial a_is and b_is
        self.append_ais_bis()
        # Tweak for polar Hydrogens
        self.polar_hydrogens()

    def extract_params_chargemol(self):
        """From Chargemol output files, extract the necessary parameters for calculation of L-J.
        Desired format:
        ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x_dipole', 'y_dipole', 'z_dipole', 'vol']
        All vals are float except atom number (int) and atom type (str).
        """

        if self.qm['ddec_version'] == 6:
            net_charge_file_name = 'DDEC6_even_tempered_net_atomic_charges.xyz'

        elif self.qm['ddec_version'] == 3:
            net_charge_file_name = 'DDEC3_net_atomic_charges.xyz'

        else:
            raise ValueError('Unsupported DDEC version; please use version 3 or 6.')

        if not exists(net_charge_file_name):
            raise FileNotFoundError('Cannot find the DDEC output file.\nThis could be indicative of several issues.\n'
                                    'Please check Chargemol is installed in the correct location and that the configs'
                                    ' point to that location.')

        with open(net_charge_file_name, 'r+') as charge_file:

            lines = charge_file.readlines()

        # Find number of atoms
        atom_total = int(lines[0])

        for pos, row in enumerate(lines):
            # Data marker:
            if 'The following XYZ' in row:
                start_pos = pos + 2
                break
        else:
            raise EOFError(f'Cannot find charge data in {net_charge_file_name}.')

        # Append the atom number and type, coords, charge, dipoles:
        for line in lines[start_pos: start_pos + atom_total]:
            a_number, a_type, *data = line.split()
            self.ddec_data.append([int(a_number), a_type] + [float(datum) for datum in data])

        charges = [atom[5] for atom in self.ddec_data]
        check_net_charge(charges, ideal_net=self.defaults_dict['charge'])

        r_cubed_file_name = 'DDEC_atomic_Rcubed_moments.xyz'

        with open(r_cubed_file_name, 'r+') as vol_file:

            lines = vol_file.readlines()

        vols = [float(line.split()[-1]) for line in lines[2:atom_total + 2]]

        for pos, atom in enumerate(self.ddec_data):
            atom.append(vols[pos])

    def extract_params_onetep(self):
        """From ONETEP output files, extract the necessary parameters for calculation of L-J.
        Desired format:
        ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'vol']
        All vals are float except atom number (int) and atom type (str).
        """

        # First file contains atom number, type and xyz coords:
        with open('ddec.xyz', 'r') as file:
            lines = file.readlines()

        for pos, line in enumerate(lines[2:]):
            atom, *coords = line.split()
            self.ddec_data.append([pos, atom] + [float(coord) for coord in coords])

        # Second file contains the rest (charges, dipoles and volumes):
        with open('ddec.onetep', 'r') as file:
            lines = file.readlines()

        charge_pos, vol_pos = False, False
        for pos, line in enumerate(lines):

            # Charges marker in file:
            if 'DDEC density' in line:
                charge_pos = pos + 7

            # Volumes marker in file:
            if 'DDEC Radial' in line:
                vol_pos = pos + 4

        if not charge_pos and vol_pos:
            raise EOFError('Cannot locate charges and / or volumes in ddec.onetep file.')

        charges = [float(line.split()[-1]) for line in lines[charge_pos: charge_pos + len(self.ddec_data)]]
        check_net_charge(charges, ideal_net=self.defaults_dict['charge'])

        # Add the AIM-Valence and the AIM-Core to get V^AIM
        volumes = [float(line.split()[2]) + float(line.split()[3]) for line in lines[vol_pos: vol_pos + len(self.ddec_data)]]

        # Add the charges and volumes to the end of the inner lists (containing coords etc)
        for pos, atom in enumerate(self.ddec_data):
            atom.extend((charges[pos], volumes[pos]))

    def append_ais_bis(self):
        """Use the AIM parameters from extract_params_*() to calculate a_i and b_i according to paper.
        Calculations from paper have been combined and simplified for faster computation.
        """

        # Beware weird units, (wrong in the paper too).
        # 'elem' : [vfree, bfree, rfree]
        # Units: [vfree: Bohr ** 3, bfree: Ha * (Bohr ** 6), rfree: Angs]

        elem_dict = {
            'H': [7.6, 6.5, 1.64],
            'C': [34.4, 46.6, 2.08],
            'N': [25.9, 24.2, 1.72],
            'O': [22.1, 15.6, 1.60],
            'F': [18.2, 9.5, 1.58],
            'S': [75.2, 134.0, 2.00],
            'Cl': [65.1, 94.6, 1.88],
            'Br': [95.7, 162.0, 1.96],
        }

        for pos, atom in enumerate(self.ddec_data):

            # r_aim = r_free * ((vol / v_free) ** (1 / 3))
            r_aim = elem_dict[f'{atom[1]}'][2] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** (1 / 3))

            # b_i = bfree * ((vol / v_free) ** 2)
            b_i = elem_dict[f'{atom[1]}'][1] * ((atom[-1] / elem_dict[f'{atom[1]}'][0]) ** 2)

            a_i = 32 * b_i * (r_aim ** 6)

            self.ddec_data[pos] += [r_aim, b_i, a_i]

    def polar_hydrogens(self):
        """Identifies the polar Hydrogens and changes the a_i, b_i values accordingly.
        May be removed / heavily changed if we switch away from atom typing and use SMARTS.
        """

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

        # Find all the polar hydrogens and store their positions / atom numbers
        polars = []
        for pair in new_pairs:
            if 'O' in pair[0] or 'N' in pair[0] or 'S' in pair[0]:
                if 'H' in pair[1]:
                    polars.append(pair)
            if 'O' in pair[1] or 'N' in pair[1] or 'S' in pair[1]:
                if 'H' in pair[0]:
                    polars.append(pair)

        if polars:
            for pair in polars:
                if 'H' in pair[0] or 'H' in pair[1]:
                    if 'H' in pair[0]:
                        polar_h_pos = int(pair[0][0]) - 1
                        polar_son_pos = int(pair[1][0]) - 1
                    else:
                        polar_h_pos = int(pair[1][0]) - 1
                        polar_son_pos = int(pair[0][0]) - 1
                    # Reset the b_i for the two polar atoms (polar h and polar sulfur, oxygen or nitrogen)
                    self.ddec_data[polar_son_pos][-2] = ((self.ddec_data[polar_son_pos][-2]) ** 0.5 + (self.ddec_data[polar_h_pos][-2]) ** 0.5) ** 2
                    self.ddec_data[polar_h_pos][-2] = 0

                    # Reset the a_i for the two polar atoms using the new b_i values.
                    self.ddec_data[polar_son_pos][-1] = 32 * self.ddec_data[polar_son_pos][-2] * (self.ddec_data[polar_son_pos][-3] ** 6)
                    self.ddec_data[polar_h_pos][-1] = 0

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

        # 3. Don't do it all
        # If a symmetrical molecule appears to not be symmetrical, then there is an issue with the QM calculation.
        # Forcing symmetrisation on "almost" symmetrical molecules obfuscates their true structure.
        # When symmetrising based on nearby atoms, effects such as hydrogen bonding can be accidentally removed.
        # This is especially problematic as molecules get larger and these effects are more pronounced.
        # For example, an atom in alinine involved in the Hydrogen bonding of an alpha helix would not be the same as
        # that atom in the same position on a different alinine molecule not in an alpha helix.

        pass

    def amend_sig_eps(self):
        """Adds the sigma, epsilon terms to the ligand class object as a dictionary.
        The ligand class object (NonbondedForce) is stored as an empty dictionary until this method is called.
        """

        # TODO Add the sigma/epsilon terms after symmetry fixes.

        # Creates Nonbondedforce dict for later xml creation.
        # Format: {0: [charge, sigma, epsilon], 1: [charge, sigma, epsilon], ... }
        # This follows the usual ordering of the atoms such as in molecule.molecule.

        non_bonded_force = {}

        # Sigma: Angs -> nm
        sigma_conversion = 0.1

        # Epsilon: (Ha * (Bohr ** 6)) / (Angs ** 6) -> kJ / mol
        # (Ha * (Bohr ** 6)) / (Angs ** 6) -> Ha = * 0.529177 ** 6
        # Ha -> kcal / mol = * 627.509
        # kcal / mol -> kJ / mol = * 4.184
        # PI = 57.65240039
        epsilon_conversion = 57.65240039        # kJ/mol

        for pos, atom in enumerate(self.ddec_data):

            if atom[-1] == 0:
                sigma = epsilon = 0

            else:
                # sigma = (a_i / b_i) ** (1 / 6)
                sigma = (atom[-1] / atom[-2]) ** (1 / 6)
                sigma *= sigma_conversion

                # eps = (b_i ** 2) / (4 * a_i)
                epsilon = (atom[-2] ** 2) / (4 * atom[-1])
                epsilon *= epsilon_conversion

            non_bonded_force[pos] = [str(atom[5]), str(sigma), str(epsilon)]

        return non_bonded_force
