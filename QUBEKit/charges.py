#!/usr/bin/env python3

# TODO
#  Improved graphing; contour plot?
#  UNITS!!

from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.helpers import extract_charge_data

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


@for_all_methods(timer_logger)
class Charges:
    """
    Calculate the esp from monopoles, dipoles and quadrupoles at sample points around the atoms in a molecule.
    Plot these points on a graph, showing the atoms as x's and the sample points as dots.
    All coords (sample_coords, atom_coords, dipole coords) are numpy arrays.
    """
    def __init__(self, molecule):

        self.molecule = molecule

        self.ddec_data, self.dipole_moment_data, self.quadrupole_moment_data = extract_charge_data(self.molecule.ddec_version)

        # List of tuples where each tuple is the xyz atom coords, followed by their partial charge
        self.atom_points = []
        # List of tuples where each tuple is the xyz esp coords, followed by the esp value
        self.esp_points = []

        self.generate_esp_data_molecule()
        self.generate_atom_plot_points()
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
    def monopole_esp(q, dist):
        """
        Calculate the esp from a monopole at a given distance
        :param q: charge at atom centre
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: monopole esp value
        """
        # TODO UNITS
        return q / (4 * np.pi * dist)

    @staticmethod
    def dipole_esp(sample_coords, atom_coords, dipole_coords, dist):
        """
        Calculate the esp from a dipole at a given sample point.
        :param sample_coords: xyz coordinates of sample point
        :param atom_coords: xyz coordinates of the atom
        :param dipole_coords: xyz dipole coordinates from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: dipole esp value
        """
        return np.sum(dipole_coords.dot(sample_coords - atom_coords)) / (dist ** 3)

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
    def quadrupole_esp(sample_coords, m_tensor, dist):
        """
        Calculate the esp from a quadrupole at a given distance
        :param sample_coords: xyz coordinates of sample point
        :param m_tensor: quadrupole moment tensor calculated from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: quadrupole esp value
        """
        return (3 * sample_coords.dot(m_tensor).dot(sample_coords)) / (2 * dist ** 5)

    def generate_esp_data_atom(self, atom_index):
        """
        For a given atom at <atom_index>:
            * extract the relevant data such as partial charge
            * produce a shell of points around that atom. For each point in the shell:
                * calculate the monopole, dipole and quadrupole esp and sum them
                * create a tuple of (x, y, z, v) where xyz are the coords of the sample point and v is the total esp

        :param atom_index: The index of the atom being analysed.
        Used to extract relevant data such as partial charges
        :return: a list of all of the tuples described above
        """

        charge = self.ddec_data[atom_index].charge

        dip_data = self.dipole_moment_data[atom_index]
        dipole_coords = np.array([*dip_data.values()])

        quad_data = self.quadrupole_moment_data[atom_index]

        atom_coords = self.molecule.coords['input'][atom_index]

        min_points_per_shell = 16
        shells = 4

        sample_points = []
        for shell in range(shells):
            # (* shell) scales the number of points per shell by the radius squared.
            # This creates a more uniform sparsity
            points_in_shell = min_points_per_shell * shell
            for theta in range(points_in_shell):
                for phi in range(points_in_shell):
                    relative_sample_coords = Charges.spherical_to_cartesian((
                        (shell + 1) * 0.2,      # TODO 0.2 is some arbitrary scale factor; fix for proper units.
                        (theta / points_in_shell) * 2 * np.pi,
                        (phi / points_in_shell) * 2 * np.pi
                    ))
                    # relative coords are centred around an arbitrary sphere at (0, 0, 0).
                    # Move them to their absolute position around the atom coords
                    sample_coords = relative_sample_coords + atom_coords

                    dist = Charges.xyz_distance(sample_coords, atom_coords)

                    mono_esp = Charges.monopole_esp(charge, dist)
                    dip_esp = Charges.dipole_esp(sample_coords, atom_coords, dipole_coords, dist)

                    m_tensor = Charges.quadrupole_moment_tensor(*quad_data.values())
                    quad_esp = Charges.quadrupole_esp(sample_coords, m_tensor, dist)

                    v_total = mono_esp + dip_esp + quad_esp

                    sample_points.append((*sample_coords, v_total))

        return sample_points

    def generate_esp_data_molecule(self):
        """
        Fill esp_points list; this is a list of tuples where each tuple is an xyz coord followed by the ESP at that point.
        This can then be used to colour the point with matplotlib.
        The list of esp_points is for all atoms in self.molecule.
        """
        for atom_index in range(len(self.molecule.coords['input'])):
            self.esp_points.extend(self.generate_esp_data_atom(atom_index))

    def generate_atom_plot_points(self):
        """
        Small helper for clearer plotting (below).
        Populates the atom_points list which is just a list of tuples (x, y, z, q)
        where xyz are the atom coords and q is the atom's partial charge.
        """
        for coord, atom in zip(self.molecule.coords['input'], self.molecule.atoms):
            point = (*coord, atom.partial_charge)
            self.atom_points.append(point)

    def plot(self):
        """
        Plot the atoms of a molecule with large crosses; colour is given by their partial charge.
        Then, plot the esp at all of the sample points with small dots; colour is given by the esp magnitude.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        ax.scatter(
            xs=[i[0] for i in self.esp_points],
            ys=[i[1] for i in self.esp_points],
            zs=[i[2] for i in self.esp_points],
            c=[i[3] for i in self.esp_points],
            marker='o',
            s=1,
            alpha=0.1
        )

        ax.scatter(
            xs=[i[0] for i in self.atom_points],
            ys=[i[1] for i in self.atom_points],
            zs=[i[2] for i in self.atom_points],
            c=[i[3] for i in self.atom_points],
            marker='X',
            s=100
        )

        plt.show()
