#!/usr/bin/env python3

# TODO Improve data structures (no more lists!)

from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import CustomNamespace
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.file_handling import extract_charge_data
from QUBEKit.utils.helpers import check_net_charge, set_net

from collections import OrderedDict, namedtuple
import decimal
import os
from shutil import copy

import numpy as np


@for_all_methods(timer_logger)
class LennardJones:

    # Beware weird units, (wrong in the paper too).
    # Units: vfree: Bohr ** 3, bfree: Ha * (Bohr ** 6), rfree: Angs
    FreeParams = namedtuple('params', 'vfree bfree rfree')
    elem_dict = {
        'H': FreeParams(7.6, 6.5, 1.64),
        'B': FreeParams(46.7, 99.5, 2.08),
        'C': FreeParams(34.4, 46.6, 2.08),
        'N': FreeParams(25.9, 24.2, 1.72),
        'O': FreeParams(22.1, 15.6, 1.60),
        'F': FreeParams(18.2, 9.5, 1.58),
        'P': FreeParams(84.6, 185, 2.07),
        'S': FreeParams(75.2, 134.0, 2.00),
        'Cl': FreeParams(65.1, 94.6, 1.88),
        'Br': FreeParams(95.7, 162.0, 1.96),
        'Si': FreeParams(101.64, 305, 2.08),
    }

    def __init__(self, molecule):

        self.molecule = molecule

        if self.molecule.charges_engine == 'chargemol':
            self.ddec_data, _, _ = extract_charge_data(self.molecule.ddec_version)

        elif self.molecule.charges_engine == 'onetep':
            self.ddec_data = self.extract_params_onetep()

        else:
            raise KeyError('Invalid charges engine provided, cannot extract charges.')

        # TODO Maybe move this to run/ligand file (or just elsewhere; it seems odd to have it here)?
        # Charge check
        charges = [atom.charge for atom in self.ddec_data.values()]
        check_net_charge(charges, ideal_net=self.molecule.charge)

        self.c8_params = None

        self.non_bonded_force = {}

    def extract_params_onetep(self):
        """
        TODO Move this to file_handling.py
        From ONETEP output files, extract the necessary parameters for calculation of L-J.
        Insert data into ddec_data in standard format

        This will exclusively be used by this class.
        It will also give less information, hence why it is here, not in the helpers file.
        """

        # Just fill in None values until they are known
        ddec_data = {i: CustomNamespace(
            atomic_symbol=atom.atomic_symbol, charge=None, volume=None, r_aim=None, b_i=None, a_i=None)
            for i, atom in enumerate(self.molecule.atoms)}

        # Second file contains the rest (charges, dipoles and volumes):
        ddec_output_file = 'ddec.onetep' if os.path.exists('ddec.onetep') else 'iter_1/ddec.onetep'
        with open(ddec_output_file, 'r') as file:
            lines = file.readlines()

        charge_pos, vol_pos = None, None
        for pos, line in enumerate(lines):

            # Charges marker in file:
            if 'DDEC density' in line:
                charge_pos = pos + 7

            # Volumes marker in file:
            if 'DDEC Radial' in line:
                vol_pos = pos + 4

        if any(position is None for position in [charge_pos, vol_pos]):
            raise EOFError('Cannot locate charges and / or volumes in ddec.onetep file.')

        charges = [float(line.split()[-1])
                   for line in lines[charge_pos: charge_pos + len(self.molecule.atoms)]]

        # Add the AIM-Valence and the AIM-Core to get V^AIM
        volumes = [float(line.split()[2]) + float(line.split()[3])
                   for line in lines[vol_pos: vol_pos + len(self.molecule.atoms)]]

        for atom_index in ddec_data:
            ddec_data[atom_index].charge = charges[atom_index]
            ddec_data[atom_index].volume = volumes[atom_index]

        return ddec_data

    def extract_c8_params(self):
        """
        Extract the C8 dispersion coefficients from the MCLF calculation's output file.
        :return: c8_params ordered list of the c8 params for each atom in molecule
        """

        with open('MCLF_C8_dispersion_coefficients.xyz') as c8_file:
            lines = c8_file.readlines()
            for i, line in enumerate(lines):
                if line.startswith(' The following '):
                    lines = lines[i + 2: -2]
                    break
            else:
                raise EOFError('Cannot locate c8 parameters in file.')

            # c8 params IN ATOMIC UNITS
            self.c8_params = [float(line.split()[-1].strip()) for line in lines]

    def apply_symmetrisation(self):
        """
        Using the atoms picked out to be symmetrised:
        apply the symmetry to the charge and volume values.
        Mutates the non_bonded_force dict
        """

        atom_types = {}
        for key, val in self.molecule.atom_symmetry_classes.items():
            atom_types.setdefault(val, []).append(key)

        # Find the average charge / volume values for each sym_set.
        # A sym_set is atoms which should have the same charge / volume values (e.g. methyl H's).
        for sym_set in atom_types.values():
            charge = sum(self.ddec_data[atom].charge for atom in sym_set) / len(sym_set)
            volume = sum(self.ddec_data[atom].volume for atom in sym_set) / len(sym_set)

            # Store the new values.
            for atom in sym_set:
                self.ddec_data[atom].charge = round(charge, 6)
                self.ddec_data[atom].volume = round(volume, 6)

        # Make sure the net charge is correct for the current precision.
        charges = [atom.charge for atom in self.ddec_data.values()]
        new_charges = set_net(charges, self.molecule.charge, 6)

        # Put the new charges back into the holder.
        for atom, new_charge in zip(self.ddec_data.values(), new_charges):
            atom.charge = new_charge

    def append_ais_bis(self):
        """
        Use the AIM parameters from extract_params_*() to calculate a_i and b_i according to paper.
        Calculations from paper have been combined and simplified for faster computation.
        """

        for atom_index, atom in self.ddec_data.items():
            try:
                atomic_symbol, atom_vol = atom.atomic_symbol, atom.volume
                # r_aim = r_free * ((vol / v_free) ** (1 / 3))
                r_aim = self.elem_dict[atomic_symbol].rfree * ((atom_vol / self.elem_dict[atomic_symbol].vfree) ** (1 / 3))

                # b_i = bfree * ((vol / v_free) ** 2)
                b_i = self.elem_dict[atomic_symbol].bfree * ((atom_vol / self.elem_dict[atomic_symbol].vfree) ** 2)

                a_i = 32 * b_i * (r_aim ** 6)

            # Element not in elem_dict.
            except KeyError:
                r_aim, b_i, a_i = 0, 0, 0

            self.ddec_data[atom_index].r_aim = r_aim
            self.ddec_data[atom_index].b_i = b_i
            self.ddec_data[atom_index].a_i = a_i

    def calculate_sig_eps(self):
        """
        Adds the charge, sigma and epsilon terms to the ligand class object in a dictionary.
        The ligand class object (NonbondedForce) is stored as an empty dictionary until this method is called.
        """

        # Creates Nonbondedforce dict for later xml creation.
        # Format: {0: [charge, sigma, epsilon], 1: [charge, sigma, epsilon], ... }
        # This follows the usual ordering of the atoms such as in molecule.coords.
        for atom_index, atom in self.ddec_data.items():

            if not atom.a_i:
                sigma = epsilon = 0
            else:
                # sigma = (a_i / b_i) ** (1 / 6)
                sigma = (atom.a_i / atom.b_i) ** (1 / 6)
                sigma *= constants.SIGMA_CONVERSION

                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (atom.b_i ** 2) / (4 * atom.a_i)
                epsilon *= constants.EPSILON_CONVERSION

            self.non_bonded_force[atom_index] = [atom.charge, sigma, epsilon]

    def correct_polar_hydrogens(self):
        """
        Identifies the polar Hydrogens and changes the a_i, b_i values accordingly.
        May be removed / heavily changed if we switch away from atom typing and use SMARTS.
        """

        # Loop through pairs in topology
        # Create new pair list with the atoms
        new_pairs = []
        for pair in self.molecule.topology.edges:
            new_pairs.append((self.molecule.atoms[pair[0]], self.molecule.atoms[pair[1]]))

        # Find all the polar hydrogens and store their positions / atom numbers
        polars = []
        for pair in new_pairs:
            if 'O' == pair[0].atomic_symbol or 'N' == pair[0].atomic_symbol or 'S' == pair[0].atomic_symbol:
                if 'H' == pair[1].atomic_symbol:
                    polars.append(pair)

            if 'O' == pair[1].atomic_symbol or 'N' == pair[1].atomic_symbol or 'S' == pair[1].atomic_symbol:
                if 'H' == pair[0].atomic_symbol:
                    polars.append(pair)

        # Find square root of all b_i values so that they can be added easily according to paper's formula.
        for atom in self.ddec_data.values():
            atom.b_i = atom.b_i ** 0.5

        if polars:
            for pair in polars:
                if 'H' == pair[0].atomic_symbol or 'H' == pair[1].atomic_symbol:
                    if 'H' == pair[0].atomic_symbol:
                        polar_h_pos = pair[0].atom_index
                        polar_son_pos = pair[1].atom_index
                    else:
                        polar_h_pos = pair[1].atom_index
                        polar_son_pos = pair[0].atom_index

                    # Calculate the new b_i for the two polar atoms (polar h and polar sulfur, oxygen or nitrogen)
                    self.ddec_data[polar_son_pos].b_i += self.ddec_data[polar_h_pos].b_i
                    self.ddec_data[polar_h_pos].b_i = 0

        for atom in self.ddec_data.values():
            # Square all the b_i values again
            atom.b_i *= atom.b_i
            # Recalculate the a_is based on the new b_is
            atom.a_i = 32 * atom.b_i * (atom.r_aim ** 6)

        # Update epsilon (not sigma) according to new a_i and b_i values
        for atom_index, atom in self.ddec_data.items():

            if atom.a_i:
                # epsilon = (b_i ** 2) / (4 * a_i)
                epsilon = (atom.b_i ** 2) / (4 * atom.a_i)
                epsilon *= constants.EPSILON_CONVERSION
            else:
                epsilon, self.non_bonded_force[atom_index][1] = 0, 0

            self.non_bonded_force[atom_index] = [atom.charge, self.non_bonded_force[atom_index][1], epsilon]

    def extract_extra_sites(self):
        """
        1) Gather the extra sites from the XYZ find parent and 2 reference atoms
        2) calculate the local coords site
        3) save the charge
        4) return back to the molecule
        (users have the option to use sites or no sites this way)
        """

        # weighting arrays for the virtual sites should not be changed
        w1o, w2o, w3o = 1.0, 0.0, 0.0   # SUM SHOULD BE 1
        w1x, w2x, w3x = -1.0, 1.0, 0.0  # SUM SHOULD BE 0
        w1y, w2y, w3y = -1.0, 0.0, 1.0  # SUM SHOULD BE 0

        with open('xyz_with_extra_point_charges.xyz') as xyz_sites:
            lines = xyz_sites.readlines()

        sites = OrderedDict()
        sites_no = 0

        for i, line in enumerate(lines[2:]):
            # get the current element
            element = str(line.split()[0])

            if element != 'X':
                # search the following entries for sites connected to this atom
                for pos_site in lines[i + 3:]:
                    # Are there are no sites?
                    if str(pos_site.split()[0]) != 'X':
                        break
                    else:
                        # get the virtual site coords
                        v_pos = np.array([float(pos_site.split()[x]) for x in range(1, 4)])
                        # get parent index number for the topology network
                        parent = i - sites_no
                        # get the two closest atoms to the parent
                        closest_atoms = list(self.molecule.topology.neighbors(parent))
                        if len(closest_atoms) < 2:
                            # find another atom if we only have one
                            # dont want to get the parent as a close atom
                            for atom in list(self.molecule.topology.neighbors(closest_atoms[0])):
                                if atom not in closest_atoms and atom != parent:
                                    closest_atoms.append(atom)
                                    break

                        # Get the xyz coordinates of the reference atoms
                        parent_pos = self.molecule.coords['qm'][parent]
                        close_a = self.molecule.coords['qm'][closest_atoms[0]]
                        close_b = self.molecule.coords['qm'][closest_atoms[1]]

                        # work out the local coordinates site using rules from the OpenMM guide
                        orig = w1o * parent_pos + w2o * close_a + close_b * w3o
                        ab = w1x * parent_pos + w2x * close_a + w3x * close_b  # rb-ra
                        ac = w1y * parent_pos + w2y * close_a + w3y * close_b  # rb-ra
                        # Get the axis unit vectors
                        z_dir = np.cross(ab, ac)
                        z_dir /= np.sqrt(np.dot(z_dir, z_dir.reshape(3, 1)))
                        x_dir = ab / np.sqrt(np.dot(ab, ab.reshape(3, 1)))
                        y_dir = np.cross(z_dir, x_dir)
                        # Get the local coordinates positions
                        p1 = np.dot((v_pos - orig), x_dir.reshape(3, 1))
                        p2 = np.dot((v_pos - orig), y_dir.reshape(3, 1))
                        p3 = np.dot((v_pos - orig), z_dir.reshape(3, 1))

                        charge = decimal.Decimal(pos_site.split()[4])

                        # store the site info [(parent top no, a, b), (p1, p2, p3), charge]]
                        sites[sites_no] = [(parent, closest_atoms[0], closest_atoms[1]), (p1 * 0.1, p2 * 0.1, p3 * 0.1), charge]
                        sites_no += 1

        self.molecule.extra_sites = sites

        # get the parent non bonded values
        for site in sites.values():
            charge, sigma, eps = self.non_bonded_force[site[0][0]]
            # Change the charge on the first entry
            charge -= site[2]
            self.non_bonded_force[site[0][0]] = [charge, sigma, eps]

    def calculate_non_bonded_force(self):
        """
        Main worker method for LennardJones class.
        Calculates the a_i and b_i values;
        Calculates the sigma and epsilon values using those a_i and b_i values;
        Redistributes L-J parameters according to polar Hydrogens, then recalculates epsilon values.
        """

        # Tweak the charge and volumes for symmetry
        if self.molecule.symmetry:
            self.apply_symmetrisation()

        # Calculate initial a_is and b_is
        self.append_ais_bis()

        # Use the a_is and b_is to calculate the non_bonded_force dict
        self.calculate_sig_eps()

        # Tweak for polar Hydrogens
        # NB DISABLE FOR FORCEBALANCE
        self.correct_polar_hydrogens()

        # Find extra site positions in local coords if present and tweak the charges of the parent
        try:
            copy(
                os.path.join(self.molecule.home, '07_charges', 'xyz_with_extra_point_charges.xyz'),
                'xyz_with_extra_point_charges.xyz'
            )
            self.extract_extra_sites()
        except FileNotFoundError:
            pass

        self.molecule.NonbondedForce = self.non_bonded_force

        for atom_index, n_b_f in self.non_bonded_force.items():
            self.molecule.atoms[atom_index].partial_charge = n_b_f[0]

    def new_calculate_non_bonded_force(self):
        """
        Using the extracted C8 params, squash the 3-term potential into two terms using curvefit.
        :return:
        """

        self.extract_params_chargemol()
        self.extract_c8_params()
        # Get the a_i and b_i params
        self.append_ais_bis()

        def objective_function(t, sig, eps):
            """Parametric form of the standard L-J potential terms."""
            return 4 * eps * ((sig / t) ** 12 - (sig / t) ** 6)

        from scipy import optimize
        # import matplotlib.pyplot as plt

        r = np.array([
            1.0, 1.2, 1.4, 1.6, 1.8,
            2.0, 2.2, 2.4, 2.6, 2.8,
            3.0, 3.2, 3.4, 3.6, 3.8,
            4.0, 4.2, 4.4, 4.6, 4.8,
            5.0
        ])

        # Constrain eps and sigma values
        bounds = ([0, 0], [1, 1])

        sig_eps = []

        for atom_index, c8 in enumerate(self.c8_params):
            # c6 is equivalent to b_i ???
            c6 = self.ddec_data[atom_index].b_i
            # a is recalculated rather than using a_i from before
            ########## UNITS ##########
            a = ((c6 * r) ** 6) / 2 + ((2 * c8 * r) ** 4) / 3
            ########## UNITS ##########
            v = (a / (r ** 12)) - (c6 / (r ** 6)) - (c8 / (r ** 8))

            popt, pcov = optimize.curve_fit(objective_function, r, v, bounds=bounds)

            sig_eps.append(popt)

            # t = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
            # plt.figure(1)
            # plt.clf()
            # plt.plot(r, v, 'bx', label='data')
            # plt.plot(t, f(t, *popt), 'r-', label=f'fit: sig={popt[0]:5f}, eps={popt[1]:5f}')
            # plt.legend()
            # plt.show()

        print(sig_eps)
