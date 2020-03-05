#!/usr/bin/env python3

"""
TODO
    Is there some dimension reduction we can do?
    Can we change the way sample points are generated to cut out computation time?
        e.g.:
        Change it so sample coords are precalculated at fixed points, then scaled using vdw_radius
    Simplify minimisation space
        Are there regions we don't need to trial v-sites in?
    Add bounds to stop v-sites being put everywhere
    futureproof!
        Make the code cleaner
        Easier to swap out elements for potential ML improvements
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

        # List of tuples where each tuple is the xyz esp coords, followed by the esp value
        self.esp_points = []
        for atom_index in range(len(self.molecule.coords['qm'])):
            for points in self.generate_esp_atom(atom_index):
                self.esp_points.append(points)

        # self.plot()
        self.minimise(1)


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
    def generate_sample_points(self, vdw_radius):
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

    def generate_esp_atom(self, atom_index):
        """
        For a given atom at <atom_index>:
            * Extract the relevant data such as partial charge
            * Produce a shell of points around that atom.
            * For each point:
                * Calculate the monopole, dipole and quadrupole esp
                * Sum them
                * Create a tuple of ((x, y, z), v): xyz are the sample point coords, v is the total esp

        :param atom_index: The index of the atom being analysed.
        Used to extract relevant data such as partial charges
        :return: a list of all of the tuples described above
        """

        atomic_symbol = self.molecule.atoms[atom_index].atomic_symbol
        vdw_radius = self.vdw_radii[atomic_symbol]

        atom_coords = self.molecule.coords['qm'][atom_index]

        charge = self.ddec_data[atom_index].charge
        dip_data = self.dipole_moment_data[atom_index]
        dipole_moment = np.array([*dip_data.values()]) * BOHR_TO_ANGS

        quad_data = self.quadrupole_moment_data[atom_index]

        mono_esp_tot, dipo_esp_tot, quad_esp_tot = 0, 0, 0

        relative_sample_points = self.generate_sample_points(vdw_radius)

        sample_coords = []
        for relative_point in relative_sample_points:

            proper_point = relative_point + atom_coords

            dist = Charges.xyz_distance(proper_point, atom_coords)
            dist_vector = proper_point - atom_coords

            mono_esp = Charges.monopole_esp_one_charge(charge, dist)
            dipo_esp = Charges.dipole_esp(dist_vector, dipole_moment, dist)

            m_tensor = Charges.quadrupole_moment_tensor(*quad_data.values())
            quad_esp = Charges.quadrupole_esp(dist_vector, m_tensor, dist)

            mono_esp_tot += mono_esp
            dipo_esp_tot += abs(dipo_esp)
            quad_esp_tot += abs(quad_esp)

            v_total = mono_esp + dipo_esp + quad_esp

            sample_coords.append((proper_point, v_total))

        n_points = len(sample_coords)

        print(f'{atomic_symbol}  ', end=' ')
        print(f'M: {(mono_esp_tot / (n_points * ANGS_TO_M)) * J_TO_KCAL_P_MOL: .6f}  ', end=' ')
        print(f'D: {(dipo_esp_tot / (n_points * ANGS_TO_M)) * J_TO_KCAL_P_MOL: .6f}  ', end=' ')
        print(f'Q: {(quad_esp_tot / (n_points * ANGS_TO_M)) * J_TO_KCAL_P_MOL: .6f}  ')

        return sample_coords

    def generate_atom_mono_esp_two_charges(self, atom_index, site_charge, site_pos):

        atomic_symbol = self.molecule.atoms[atom_index].atomic_symbol
        vdw_radius = self.vdw_radii[atomic_symbol]

        atom_coords = self.molecule.coords['qm'][atom_index]

        charge = self.ddec_data[atom_index].charge - site_charge

        mono_esp_tot = 0

        relative_sample_points = self.generate_sample_points(vdw_radius)

        sample_coords = []
        for relative_point in relative_sample_points:
            proper_point = relative_point + atom_coords

            dist = Charges.xyz_distance(proper_point, atom_coords)
            site_dist = Charges.xyz_distance(proper_point, site_pos)
            mono_esp = Charges.monopole_esp_charges(charge, site_charge, dist, site_dist)

            mono_esp_tot += mono_esp
            sample_coords.append((proper_point, mono_esp))

        n_points = len(sample_coords)

        print(f'{atomic_symbol}  ', end=' ')
        print(f'M: {(mono_esp_tot / (n_points * ANGS_TO_M)) * J_TO_KCAL_P_MOL: .6f}  ')

        return sample_coords

    def minimise(self, atom_index):

        # This is what we're aiming for
        v_no_off_site = [v for coords, v in self.generate_esp_atom(atom_index)]

        # Change q, x, y, z and minimise sum(abs(v_no_off_site - mono_esp))
        def objective_function(q=0, x=0, y=0, z=0):
            total = 0
            for v, m in zip(v_no_off_site, [v for _, v in self.generate_atom_mono_esp_two_charges(atom_index, q, np.array([x, y, z]))]):
                total += abs((v - m))
            return total

        # Bounds should be set so that there's a max distance from the atom for the v-site
        # and it's charge is no greater than a certain amount

        return minimize(objective_function, np.array([0, 0, 0, 0]))

    def plot(self):
        """
        Plot the atoms of a molecule with large crosses; colour is given by their partial charge.
        Then, plot the esp at all of the sample points with small dots; colour is given by the esp magnitude.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        # TODO Use numpy to rotate esp_points matrix for faster variable access.
        ax.scatter(
            xs=[i[0][0] for i in self.esp_points],
            ys=[i[0][1] for i in self.esp_points],
            zs=[i[0][2] for i in self.esp_points],
            c=[i[1] for i in self.esp_points],
            marker='o',
            s=2,
            alpha=0.5
        )

        ax.scatter(
            xs=[i[0][0] for i in self.atom_points],
            ys=[i[0][1] for i in self.atom_points],
            zs=[i[0][2] for i in self.atom_points],
            c=[i[1] for i in self.atom_points],
            marker='X',
            s=100
        )

        plt.show()


class FitCharges:
    """
    Using the sample points generated from the Charges class:
        * Minimise average_over_points(abs(Vmm-Vqm))
        * Move the off-site charge to its optimised position.

    """
    pass
