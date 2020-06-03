#!/usr/bin/env python3

"""
TODO
    Convert sample_points to a matrix.
        Cleaner since you can just broadcast the offset rather than loop over the list
    There's still a little bit of repetition
    Plotting is a bit gross (probably only used for testing though?)
    Make it easier to swap out elements for potential ML improvements
"""

from QUBEKit.utils.constants import ANGS_TO_M, BOHR_TO_ANGS, ELECTRON_CHARGE, J_TO_KCAL_P_MOL, PI, VACUUM_PERMITTIVITY
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.file_handling import extract_charge_data

from functools import lru_cache
import math

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize


@for_all_methods(timer_logger)
class Charges:
    """
    Calculate the esp from monopoles, dipoles and quadrupoles at sample points around the atoms in a molecule.
    Plot these points on a graph, showing the atoms as crosses (x) and the sample points as dots (o).
    Numpy arrays are used throughout for faster calculation of esp values.
    """

    # van der Waal's radii of atoms common in organic compounds; units: Angstroms
    vdw_radii = {
        'H': 1.44,
        'B': 2.04,
        'C': 1.93,
        'N': 1.83,
        'O': 1.75,
        'F': 1.68,
        'P': 2.07,
        'S': 2.02,
        'Cl': 1.97,
        'I': 2.25,
    }

    def __init__(self, molecule):

        self.molecule = molecule

        self.ddec_data, self.dipole_moment_data, self.quadrupole_moment_data = extract_charge_data(self.molecule.ddec_version)

        # TODO Convert to numpy arrays!
        # List of tuples where each tuple is the xyz atom coords, followed by their partial charge
        self.atom_points = [(coord, atom.partial_charge)       # ((x, y, z), q)
                            for coord, atom in zip(self.molecule.coords['qm'], self.molecule.atoms)]

        for atom_index, atom in enumerate(self.molecule.atoms):
            # TODO Extend allowed element types
            if atom.atomic_symbol in ['F', 'Cl', 'Br', 'O', 'N']:
                self.sample_points = self.generate_sample_points_atom(atom_index)
                self.v_site_coords = self.fit(atom_index)
                self.plot()

    @staticmethod
    def spherical_to_cartesian(spherical_coords):
        """
        :return: Cartesian (x, y, z) coords from the spherical (r, theta, phi) coords.
        """
        r, theta, phi = spherical_coords
        return np.array([r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)])

    @staticmethod
    def xyz_distance(point1, point2):
        """
        :param point1: coordinates of a point
        :param point2: coordinates of another point
        :return: distance between the two points
        """
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def monopole_esp_one_charge(charge, dist):
        """
        Calculate the esp from a monopole at a given distance
        :param charge: charge at atom centre
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: monopole esp value
        """
        return (charge * ELECTRON_CHARGE * ELECTRON_CHARGE) / (
                4 * PI * VACUUM_PERMITTIVITY * dist)

    @staticmethod
    def monopole_esp_charges(charge1, charge2, dist1, dist2):
        return ((ELECTRON_CHARGE * ELECTRON_CHARGE) / (4 * PI * VACUUM_PERMITTIVITY)) * (charge1 / dist1 + charge2 / dist2)

    @staticmethod
    def dipole_esp(dist_vector, dipole_moment, dist):
        """
        Calculate the esp from a dipole at a given sample point.
        :param dist_vector: atom_coords - sample_coords
        :param dipole_moment: dipole moment xyz components from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: dipole esp value
        """
        return np.sum((dipole_moment * ELECTRON_CHARGE * ELECTRON_CHARGE).dot(dist_vector)) / (
                4 * PI * VACUUM_PERMITTIVITY * dist ** 3)

    @staticmethod
    def quadrupole_moment_tensor(q_xy, q_xz, q_yz, q_x2_y2, q_3z2_r2):
        """
        :params: quadrupole moment components from Chargemol output
        :return: quadrupole moment tensor, M
        """
        return np.array([
            [q_x2_y2 / 2 - q_3z2_r2 / 6, q_xy, q_xz],
            [q_xy, -q_x2_y2 / 2 - q_3z2_r2 / 6, q_yz],
            [q_xz, q_yz, q_3z2_r2 / 3]
        ])

    @staticmethod
    def quadrupole_esp(dist_vector, m_tensor, dist):
        """
        Calculate the esp from a quadrupole at a given distance
        :param dist_vector: atom_coords - sample_coords
        :param m_tensor: quadrupole moment tensor calculated from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: quadrupole esp value
        """
        return (3 * ELECTRON_CHARGE * ELECTRON_CHARGE * dist_vector.dot(m_tensor * (BOHR_TO_ANGS ** 2)).dot(dist_vector)) / (
                8 * PI * VACUUM_PERMITTIVITY * dist ** 5)

    @lru_cache(maxsize=None)
    def generate_sample_points_relative(self, vdw_radius):
        """
        Generate evenly distributed points in a series of shells around the point (0, 0, 0)
        This uses fibonacci spirals to produce an even spacing of points on a sphere.
        """

        min_points_per_shell = 16
        shells = 5
        increment = PI * (3 - math.sqrt(5))

        relative_sample_points = []
        for shell in range(1, shells + 1):
            points_in_shell = min_points_per_shell * shell ** 2

            for i in range(points_in_shell):
                y = ((i * 2) / points_in_shell - 1) + 1 / points_in_shell
                r = math.sqrt(1 - pow(y, 2)) * shell * vdw_radius

                phi = ((i + 1) % points_in_shell) * increment

                x = np.cos(phi) * r
                z = np.sin(phi) * r

                relative_sample_points.append(np.array([x, y, z]))

        return relative_sample_points

    def generate_sample_points_atom(self, atom_index):
        """
        * Get the vdw radius of the atom which is being analysed
        * Using the relative sample points generated from generate_sample_points_relative():
            * Offset all of the points by the position of the atom coords
        :param atom_index: index of the atom around which a v-site will be fit
        :return: a list of all the sample points (length 3 np arrays) to be used in the
        fitting of the v-site.
        """

        atom = self.molecule.atoms[atom_index]
        atom_coords = self.molecule.coords['qm'][atom_index]
        vdw_radius = self.vdw_radii[atom.atomic_symbol]

        sample_points = self.generate_sample_points_relative(vdw_radius)
        for point in sample_points:
            point += atom_coords

        return sample_points

    def generate_esp_atom(self, atom_index):
        """
        Generate an ordered list of ESP values at each sample point around the atom at <atom_index>
        ESP is calculated using the monopole, dipole and quadrupole.
        :param atom_index: The index of the atom being analysed.
        Used to extract relevant data such as partial charges
        :return: list of ESP values at each point in the list of relative sample points
        This is the list to fit TO.
        """

        atom_coords = self.molecule.coords['qm'][atom_index]

        charge = self.ddec_data[atom_index].charge
        dip_data = self.dipole_moment_data[atom_index]
        dipole_moment = np.array([*dip_data.values()]) * BOHR_TO_ANGS

        quad_data = self.quadrupole_moment_data[atom_index]

        no_site_esps = []
        for point in self.sample_points:
            dist = Charges.xyz_distance(point, atom_coords)
            dist_vector = point - atom_coords

            mono_esp = Charges.monopole_esp_one_charge(charge, dist)
            dipo_esp = Charges.dipole_esp(dist_vector, dipole_moment, dist)

            m_tensor = Charges.quadrupole_moment_tensor(*quad_data.values())
            quad_esp = Charges.quadrupole_esp(dist_vector, m_tensor, dist)

            # TODO Should be abs?
            # v_total = mono_esp + abs(dipo_esp) + abs(quad_esp)
            v_total = mono_esp + dipo_esp + quad_esp
            no_site_esps.append((v_total / ANGS_TO_M) * J_TO_KCAL_P_MOL)

        return no_site_esps

    def generate_atom_mono_esp_two_charges(self, atom_index, site_charge, site_coords):
        """
        Generate an ordered list of ESP values at each sample point around the atom at <atom_index>
        ESP is calculated using just the monopole from the charge on the atom and the virtual site.
        :param atom_index: The index of the atom being analysed.
        :param site_charge: The charge of the virtual site.
        :param site_coords: The coordinates of the virtual site.
        :return: list of ESP values at each point in the list of relative sample points.
        This is the list of values which will be adjusted to fit to the values from generate_esp_atom()
        """

        atom_coords = self.molecule.coords['qm'][atom_index]
        # New charge of the atom, having removed the v-site's charge.
        atom_charge = self.ddec_data[atom_index].charge - site_charge

        # TODO Just append q to each item in self.sample_points (cleaner)
        v_site_esps = []
        for point in self.sample_points:
            dist = Charges.xyz_distance(point, atom_coords)
            site_dist = Charges.xyz_distance(point, site_coords)

            mono_esp = Charges.monopole_esp_charges(atom_charge, site_charge, dist, site_dist)
            v_site_esps.append((mono_esp / ANGS_TO_M) * J_TO_KCAL_P_MOL)

        return v_site_esps

    def plot(self):
        """
        Plot the coordinates of the atoms and the virtual site.
        The atoms are crosses, the virtual site is a dot.
        Colour represents the charge of the particles.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        # Atoms and their charges
        ax.scatter(
            xs=[i[0][0] for i in self.atom_points],
            ys=[i[0][1] for i in self.atom_points],
            zs=[i[0][2] for i in self.atom_points],
            c=[i[1] for i in self.atom_points],
            marker='X',
            s=100
        )

        # V-site and its charge
        ax.scatter(
            xs=[self.v_site_coords[0][0]],
            ys=[self.v_site_coords[0][1]],
            zs=[self.v_site_coords[0][2]],
            c=[self.v_site_coords[1]],
            marker='o',
            s=100
        )

        plt.show()

    def get_vector_from_coords(self, atom_index):
        """
        Given the coords of the atom which will have a v-site, as well as its neighbouring atom(s) coords:
            return the vector along which the v-site will sit.
        For halogens, this is linear from C-Halo-Site.
        For oxygen, this bisects the atom-oxygen-atom angle and points away from the two bonds. (like trigonal planar).
        For nitrogen, this bisects the nitrogen-atom^3 and points away from the three bonds. (like tetrahedal).
        :param atom_index: The index of the atom being analysed.
        :return the vector along which the v-site will sit.
        """

        # TODO rewrite; this is gross.
        atom = self.molecule.atoms[atom_index]
        atom_coords = self.molecule.coords['qm'][atom_index]
        if atom.atomic_symbol in ['F', 'Cl', 'Br']:
            bonded_index = atom.bonds[0]    # [0] is used since bonds is a one item list
            bonded_coords = self.molecule.coords['qm'][bonded_index]
            r_ab = bonded_coords - atom_coords
            return r_ab

        if atom.atomic_symbol in ['O']:
            bonded_index_b, bonded_index_c = atom.bonds
            bonded_coords_b = self.molecule.coords['qm'][bonded_index_b]
            bonded_coords_c = self.molecule.coords['qm'][bonded_index_c]
            r_ab = bonded_coords_b - atom_coords
            r_ac = bonded_coords_c - atom_coords
            return r_ab + r_ac

        if atom.atomic_symbol in ['N']:
            bonded_index_b, bonded_index_c, bonded_index_d = atom.bonds
            bonded_coords_b = self.molecule.coords['qm'][bonded_index_b]
            bonded_coords_c = self.molecule.coords['qm'][bonded_index_c]
            bonded_coords_d = self.molecule.coords['qm'][bonded_index_d]
            r_ab = bonded_coords_b - atom_coords
            r_ac = bonded_coords_c - atom_coords
            r_ad = bonded_coords_d - atom_coords
            return np.cross((r_ab - r_ac), (r_ad - r_ac))

    def esp_from_lambda_and_charge(self, atom_index, q, lam, vec):
        """
        :param atom_index: index of the atom with a virtual site to be fit to
        :param q: charge of the virtual site
        :param lam: scaling of the vector along which the v-site sits
        :param vec: the vector along which the v-site sits
        Place a v-site at the correct position along the vector by scaling according to the lambda
        calculate the esp from the atom and the v-site.
        """

        # This is the current position of the v-site (moved by the fit() method)
        site_coords = (-vec * lam) + self.molecule.coords['qm'][atom_index]
        return self.generate_atom_mono_esp_two_charges(atom_index, q, site_coords)

    def fit(self, atom_index):
        """
        Calculate the "ideal" esp using the more precise monopole + dipole + quadrupole method.
        Calculate the vector along which a v-site would sit.
        Fit a monopole esp to the "ideal" esp by varying the charge and position of the v-site along the vector
        :param atom_index: the atom which will have a v-site (halogen, oxygen, nitrogen)
        :return: The scale factor (lambda) of the vector, and q, the charge of the v-site
        """

        no_site_esps = self.generate_esp_atom(atom_index)
        # Vector along which the v-site will sit.
        vec = self.get_vector_from_coords(atom_index)

        def objective_function(q_lam):
            """
            Error is defined as the sum of the absolute differences between the esp at each point,
            calculated with and without a virtual site.
            """
            site_esps = self.esp_from_lambda_and_charge(atom_index, *q_lam, vec)
            error = sum(abs(no_site_esp - site_esp)
                        for no_site_esp, site_esp in zip(no_site_esps, site_esps))
            print(error)
            return error

        # TODO Are these acceptable bounds? Should they be determined through the vdW radii? (yes)
        bounds = ((-0.5, 0.5), (0.1, 1.0))

        result = minimize(objective_function, np.array([0, 1]), bounds=bounds)
        q, lam = result.x
        v_site_coords = (-vec * lam) + self.molecule.coords['qm'][atom_index]

        print(f'coords: {v_site_coords} charge: {q} scale factor, lambda: {lam}')
        # TODO Use the coords and charge to add a virtual site to the xml for the molecule.
        return v_site_coords, q
