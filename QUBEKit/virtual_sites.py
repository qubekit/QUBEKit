#!/usr/bin/env python3

from QUBEKit.utils.constants import BOHR_TO_ANGS, ELECTRON_CHARGE, J_TO_KCAL_P_MOL, M_TO_ANGS, PI, VACUUM_PERMITTIVITY
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.file_handling import extract_charge_data

from functools import lru_cache

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

# DO NOT REMOVE THIS IMPORT. ALTHOUGH IT IS NOT EXPLICITLY CALLED, IT IS NEEDED FOR 3D PLOTTING.
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from scipy.optimize import minimize


@for_all_methods(timer_logger)
class Charges:
    """
    * Identify atoms which need a v-site.
    * Generate sample points in shells around that atom (shells are 1.4-2.0x the vdW radius).
    * Calculate the multipole expansion esp at all of those sample points.
    * Identify the vectors along which a single virtual site would sit, and two virtual sites would sit.
    * Move the virtual sites along this vector and vary the charges.
    * Calculate the monopole esp at all the sample points with each move.
    * Fit the positions and charges of the virtual sites, minimising the difference between the
    full multipole esp and the monopole esp with a virtual site.
    * Store the final locations and charges of the virtual sites, as well as the errors.
    * Plot the results

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
        self.coords = self.molecule.coords['qm'] if self.molecule.coords['qm'] is not [] else self.molecule.coords['input']

        self.ddec_data, self.dipole_moment_data, self.quadrupole_moment_data = extract_charge_data(self.molecule.ddec_version)

        # List of tuples where each tuple is the xyz atom coords, followed by their partial charge
        self.atom_points = [(coord, atom.partial_charge)       # [((x, y, z), q), ... ]
                            for coord, atom in zip(self.coords, self.molecule.atoms)]

        # List of tuples where each tuple if the xyz coords of the v-site(s), followed by their charge
        self.v_sites_coords = []        # [((x, y, z), q), ... ]

        # Kept separate for graphing comparisons
        self.one_site_coords = None     # [((x, y, z), q), ... ]
        self.two_site_coords = None     # [((x, y, z), q), ... ]

        self.site_errors = {
            0: None,
            1: None,
            2: None,
        }

        for atom_index, atom in enumerate(self.molecule.atoms):
            if atom.atomic_symbol in ['N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']:
                self.sample_points = self.generate_sample_points_atom(atom_index)
                self.no_site_esps = self.generate_esp_atom(atom_index)
                self.fit(atom_index)
                self.plot()
                self.write_xyz()

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
    def monopole_esp_two_charges(charge1, charge2, dist1, dist2):
        """
        Calculate the esp from a monopole with two charges, each a different distance from the point of measurement
        :return: monopole esp value
        """
        return ((ELECTRON_CHARGE * ELECTRON_CHARGE) / (4 * PI * VACUUM_PERMITTIVITY)) * (
                charge1 / dist1 + charge2 / dist2)

    @staticmethod
    def monopole_esp_three_charges(charge1, charge2, charge3, dist1, dist2, dist3):
        """
        Calculate the esp from a monopole with three charges, each a different distance from the point of measurement
        :return: monopole esp value
        """
        return ((ELECTRON_CHARGE * ELECTRON_CHARGE) / (4 * PI * VACUUM_PERMITTIVITY)) * (
                charge1 / dist1 + charge2 / dist2 + charge3 / dist3)

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
        return (dipole_moment * ELECTRON_CHARGE * ELECTRON_CHARGE).dot(dist_vector) / (
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

        radius of points are between 1.4-2.0x the vdW radius
        :return: list of numpy arrays where each array is the xyz coordinates of a sample point.
        """

        min_points_per_shell = 32
        shells = 5
        phi = PI * (3.0 - np.sqrt(5.0))

        relative_sample_points = []
        for shell in range(shells):
            shell += 1
            points_in_shell = min_points_per_shell * shell * shell
            # 1.4-2.0x the vdw_radius
            shell_radius = (1.4 + ((2.0 - 1.4) / shells) * shell) * vdw_radius

            for i in range(points_in_shell):
                y = 1 - (i / (points_in_shell - 1)) * 2
                y_rad = np.sqrt(1 - y * y) * shell_radius
                y *= shell_radius

                theta = i * phi

                x = np.cos(theta) * y_rad
                z = np.sin(theta) * y_rad

                relative_sample_points.append(np.array([x, y, z]))

        return relative_sample_points

    def generate_sample_points_atom(self, atom_index):
        """
        * Get the vdw radius of the atom which is being analysed
        * Using the relative sample points generated from generate_sample_points_relative():
            * Offset all of the points by the position of the atom coords
        :param atom_index: index of the atom around which a v-site will be fit
        :return: list of numpy arrays where each array is the xyz coordinates of a sample point.
        """

        atom = self.molecule.atoms[atom_index]
        atom_coords = self.coords[atom_index]
        vdw_radius = self.vdw_radii[atom.atomic_symbol]

        sample_points = self.generate_sample_points_relative(vdw_radius)
        for point in sample_points:
            point += atom_coords

        return sample_points

    def generate_esp_atom(self, atom_index):
        """
        Using the multipole expansion, calculate the esp at each sample point around an atom.
        :param atom_index: The index of the atom being analysed.
        :return: Ordered list of esp values at each sample point around the atom.
        """

        atom_coords = self.coords[atom_index]

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

            v_total = (mono_esp + dipo_esp + quad_esp) * M_TO_ANGS * J_TO_KCAL_P_MOL
            no_site_esps.append(v_total)

        return no_site_esps

    def generate_atom_mono_esp_two_charges(self, atom_index, site_charge, site_coords):
        """
        With a virtual site, calculate the monopole esp at each sample point around an atom.
        :param atom_index: The index of the atom being analysed.
        :param site_charge: The charge of the virtual site.
        :param site_coords: numpy array of the xyz position of the virtual site.
        :return: Ordered list of esp values at each sample point around the atom.
        """

        atom_coords = self.coords[atom_index]
        # New charge of the atom, having removed the v-site's charge.
        atom_charge = self.ddec_data[atom_index].charge - site_charge

        v_site_esps = []
        for point in self.sample_points:
            dist = Charges.xyz_distance(point, atom_coords)
            site_dist = Charges.xyz_distance(point, site_coords)

            mono_esp = Charges.monopole_esp_two_charges(atom_charge, site_charge, dist, site_dist)
            v_site_esps.append(mono_esp * M_TO_ANGS * J_TO_KCAL_P_MOL)

        return v_site_esps

    def generate_atom_mono_esp_three_charges(self, atom_index, q_a, q_b, site_a_coords, site_b_coords):
        """
        Calculate the esp at each sample point when two virtual sites are placed around an atom.
        :param atom_index: The index of the atom being analysed.
        :param q_a: charge of v-site a
        :param q_b: charge of v-site b
        :param site_a_coords: coords of v-site a
        :param site_b_coords: coords of v-site b
        :return: ordered list of esp values at each sample point
        """

        atom_coords = self.coords[atom_index]
        atom_charge = self.ddec_data[atom_index].charge - (q_a + q_b)

        v_site_esps = []
        for point in self.sample_points:
            dist = Charges.xyz_distance(point, atom_coords)
            site_a_dist = Charges.xyz_distance(point, site_a_coords)
            site_b_dist = Charges.xyz_distance(point, site_b_coords)

            mono_esp = Charges.monopole_esp_three_charges(atom_charge, q_a, q_b, dist, site_a_dist, site_b_dist)
            v_site_esps.append(mono_esp * M_TO_ANGS * J_TO_KCAL_P_MOL)

        return v_site_esps

    def get_vector_from_coords(self, atom_index, n_sites=1, alt=False):
        """
        Given the coords of the atom which will have a v-site and its neighbouring atom(s) coords,
        calculate the vector along which the virtual site will sit.
        :param atom_index: The index of the atom being analysed.
        :param n_sites: The number of virtual sites being placed around the atom.
        :param alt: When placing two sites on an atom with two bonds, there are two placements.
            Is this the usual placement, or the alternative (rotated 90 degrees around the bisecting vector).
        :return Vector(s) along which the v-site will sit. (np array)
        """

        atom = self.molecule.atoms[atom_index]
        atom_coords = self.coords[atom_index]

        # e.g. halogens
        if len(atom.bonds) == 1:
            bonded_index = atom.bonds[0]    # [0] is used since bonds is a one item list
            bonded_coords = self.coords[bonded_index]
            r_ab = atom_coords - bonded_coords
            if n_sites == 1:
                return r_ab
            return r_ab, r_ab

        # e.g. oxygen
        if len(atom.bonds) == 2:
            bonded_index_b, bonded_index_c = atom.bonds
            bonded_coords_b = self.coords[bonded_index_b]
            bonded_coords_c = self.coords[bonded_index_c]
            r_ab = atom_coords - bonded_coords_b
            r_ac = atom_coords - bonded_coords_c
            if n_sites == 1:
                return r_ab + r_ac
            if alt:
                return (r_ab + r_ac), np.cross(r_ab, r_ac)
            return (r_ab + r_ac), np.cross((r_ab + r_ac), np.cross(r_ab, r_ac))

        # e.g. nitrogen
        if len(atom.bonds) == 3:
            bonded_index_b, bonded_index_c, bonded_index_d = atom.bonds
            bonded_coords_b = self.coords[bonded_index_b]
            bonded_coords_c = self.coords[bonded_index_c]
            bonded_coords_d = self.coords[bonded_index_d]
            r_vec = np.cross((bonded_coords_b - bonded_coords_c), (bonded_coords_d - bonded_coords_c))
            if n_sites == 1:
                return r_vec
            else:
                if atom.atomic_symbol == 'N':
                    h_s = []
                    for atom_index in atom.bonds:
                        if self.molecule.atoms[atom_index].atomic_symbol == 'H':
                            h_s.append(atom_index)
                    # Special case (amine group); position is slightly different
                    if len(h_s) == 2:
                        h_a_coords = self.coords[h_s[0]]
                        h_b_coords = self.coords[h_s[1]]
                        r_ha = atom_coords - h_a_coords
                        r_hb = atom_coords - h_b_coords

                        return r_vec, r_ha + r_hb
                return r_vec, r_vec

    def esp_from_lambda_and_charge(self, atom_index, q, lam, vec):
        """
        Place a v-site at the correct position along the vector by scaling according to the lambda
        calculate the esp from the atom and the v-site.
        :param atom_index: index of the atom with a virtual site to be fit to
        :param q: charge of the virtual site
        :param lam: scaling of the vector along which the v-site sits
        :param vec: the vector along which the v-site sits
        :return: Ordered list of esp values at each sample point
        """

        # This is the current position of the v-site (moved by the fit() method)
        site_coords = (vec * lam) + self.coords[atom_index]
        return self.generate_atom_mono_esp_two_charges(atom_index, q, site_coords)

    def sites_coords_from_vecs_and_lams(self, atom_index, lam_a, lam_b, vec_a, vec_b):
        """
        Get the two virtual site coordinates from the vectors they sit along and the atom they are attached to.
        :param atom_index: The index of the atom being analysed.
        :param lam_a: scale factor for vec_a
        :param lam_b: scale factor for vec_b
        :param vec_a: vector deciding virtual site position
        :param vec_b: vector deciding virtual site position
        :return: tuple of np arrays which are the xyz coordinates of the v-sites
        """

        if len(self.molecule.atoms[atom_index].bonds) == 2:
            site_a_coords = (vec_a * lam_a) + (vec_b * lam_b) + self.coords[atom_index]
            site_b_coords = (vec_a * lam_a) - (vec_b * lam_b) + self.coords[atom_index]
        else:
            site_a_coords = (vec_a * lam_a) + self.coords[atom_index]
            site_b_coords = (vec_b * lam_b) + self.coords[atom_index]

        return site_a_coords, site_b_coords

    def esp_from_lambdas_and_charges(self, atom_index, q_a, q_b, lam_a, lam_b, vec_a, vec_b):
        """
        Place v-sites at the correct positions along the vectors by scaling according to the lambdas
        calculate the esp from the atom and the v-sites.
        :param atom_index: The index of the atom being analysed.
        :param q_a: charge of v-site a
        :param q_b: charge of v-site b
        :param lam_a: scale factor for vec_a
        :param lam_b: scale factor for vec_b
        :param vec_a: vector deciding virtual site position
        :param vec_b: vector deciding virtual site position
        :return: Ordered list of esp values at each sample point
        """

        site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(atom_index, lam_a, lam_b, vec_a, vec_b)

        return self.generate_atom_mono_esp_three_charges(atom_index, q_a, q_b, site_a_coords, site_b_coords)

    def fit(self, atom_index, max_err=1.005):
        """
        * Take the atom which will have a v-site fit around it
        * Calculate all possible vectors depending on 1 site, 2 site, rot by 90 deg etc
        * Fit
        * Store all v-site coords (one site, two sites)
        * Which fit had lowest error?
        :param max_err: If the addition of a v-site only reduces the error by a factor of max_err, ignore it.
        :param atom_index: The index of the atom being analysed.
        """

        def one_site_objective_function(q_lam, vec):
            site_esps = self.esp_from_lambda_and_charge(atom_index, *q_lam, vec)
            error = sum(abs(no_site_esp - site_esp)
                        for no_site_esp, site_esp in zip(self.no_site_esps, site_esps))
            return error

        def two_sites_objective_function(q_q_lam_lam, vec_a, vec_b):
            site_esps = self.esp_from_lambdas_and_charges(atom_index, *q_q_lam_lam, vec_a, vec_b)
            error = sum(abs(no_site_esp - site_esp)
                        for no_site_esp, site_esp in zip(self.no_site_esps, site_esps))
            return error
        
        # q_a, q_b, lam_a, lam_b
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (0.01, 0.5), (0.01, 0.5))
        n_sample_points = len(self.no_site_esps)

        # No site
        vec = self.get_vector_from_coords(atom_index, n_sites=1)
        no_site_error = one_site_objective_function((0, 1), vec)
        self.site_errors[0] = no_site_error / n_sample_points

        # One site
        one_site_fit = minimize(one_site_objective_function, np.array([0, 1]), args=vec, bounds=bounds[1:3])
        self.site_errors[1] = one_site_fit.fun / n_sample_points
        q, lam = one_site_fit.x
        one_site_coords = [((vec * lam) + self.coords[atom_index], q)]
        self.one_site_coords = one_site_coords

        # Two sites (first orientation)
        vec_a, vec_b = self.get_vector_from_coords(atom_index, n_sites=2)
        two_site_fit = minimize(two_sites_objective_function, np.array([0, 0, 1, 1]), args=(vec_a, vec_b), bounds=bounds)
        self.site_errors[2] = two_site_fit.fun / n_sample_points
        q_a, q_b, lam_a, lam_b = two_site_fit.x
        site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(atom_index, lam_a, lam_b, vec_a, vec_b)
        two_site_coords = [(site_a_coords, q_a), (site_b_coords, q_b)]
        self.two_site_coords = two_site_coords

        # Two sites (alternative orientation)
        if len(self.molecule.atoms[atom_index].bonds) == 2:
            vec_a, vec_b = self.get_vector_from_coords(atom_index, n_sites=2, alt=True)
            alt_two_site_fit = minimize(two_sites_objective_function, np.array([0, 0, 1, 1]), args=(vec_a, vec_b), bounds=bounds)
            self.site_errors[2] = alt_two_site_fit.fun / n_sample_points
            q_a, q_b, lam_a, lam_b = alt_two_site_fit.x
            site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(atom_index, lam_a, lam_b, vec_a, vec_b)
            alt_two_site_coords = [(site_a_coords, q_a), (site_b_coords, q_b)]
            self.two_site_coords = alt_two_site_coords

        if self.site_errors[0] < min(self.site_errors[1] * max_err, self.site_errors[2] * max_err):
            print('No virtual site placement has reduced the error significantly.')
        elif self.site_errors[1] < self.site_errors[2] * max_err:
            print('The addition of one virtual site was found to be best.')
            self.v_sites_coords.extend(self.one_site_coords)
        else:
            print('The addition of two virtual sites was found to be best.')
            self.v_sites_coords.extend(self.two_site_coords)

        print(
            f'Errors (kcal/mol):\n'
            f'No Site     One Site     Two Sites\n'
            f'{self.site_errors[0]:.4f}      {self.site_errors[1]:.4f}       {self.site_errors[2]:.4f}'
        )

    def plot(self):
        """
        Figure with three subplots.
        All plots show the atoms and bonds as balls and sticks; virtual sites are x's; sample points are dots.
            * Plot showing the positions of the sample points.
            * Plot showing the position of a single virtual site.
            * Plot showing the positions of two virtual sites.
        Errors are included to show the impact of virtual site placements.
        """

        fig = plt.figure(figsize=plt.figaspect(0.33), tight_layout=True)
        # fig.suptitle('Virtual Site Placements', fontsize=20)

        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        cmap = 'cool'

        samp_plt = fig.add_subplot(1, 3, 1, projection='3d')
        one_plt = fig.add_subplot(1, 3, 2, projection='3d')
        two_plt = fig.add_subplot(1, 3, 3, projection='3d')

        plots = [samp_plt, one_plt, two_plt]
        # Add atom positions to all subplots
        for plot in plots:
            plot.scatter(
                xs=[i[0][0] for i in self.atom_points],
                ys=[i[0][1] for i in self.atom_points],
                zs=[i[0][2] for i in self.atom_points],
                c=[i[1] for i in self.atom_points],
                marker='o',
                s=200,
                cmap=cmap,
                norm=norm,
            )

            # Plot the bonds as connecting lines
            for bond in self.molecule.topology.edges:
                plot.plot(
                    xs=[self.coords[bond[0]][0], self.coords[bond[1]][0]],
                    ys=[self.coords[bond[0]][1], self.coords[bond[1]][1]],
                    zs=[self.coords[bond[0]][2], self.coords[bond[1]][2]],
                    c='darkslategrey',
                    alpha=0.5
                )

        # Left subplot contains the sample point positions
        samp_plt.scatter(
            xs=[i[0] for i in self.sample_points],
            ys=[i[1] for i in self.sample_points],
            zs=[i[2] for i in self.sample_points],
            c='darkslategrey',
            marker='o',
            s=5
        )
        samp_plt.title.set_text(f'Sample Points Positions\nError: {self.site_errors[0]: .5}')

        # Centre subplot contains the single v-site
        one_plt.scatter(
            xs=[i[0][0] for i in self.one_site_coords],
            ys=[i[0][1] for i in self.one_site_coords],
            zs=[i[0][2] for i in self.one_site_coords],
            c=[i[1] for i in self.one_site_coords],
            marker='x',
            s=200,
            cmap=cmap,
            norm=norm,
        )
        one_plt.title.set_text(f'One Site Position\nError: {self.site_errors[1]: .5}')

        # Right subplot contains the two v-sites
        two_plt.scatter(
            xs=[i[0][0] for i in self.two_site_coords],
            ys=[i[0][1] for i in self.two_site_coords],
            zs=[i[0][2] for i in self.two_site_coords],
            c=[i[1] for i in self.two_site_coords],
            marker='x',
            s=200,
            cmap=cmap,
            norm=norm,
        )
        error = self.site_errors[2]
        two_plt.title.set_text(f'Two Sites Positions\nError: {error: .5}')

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm)
        cbar.ax.set_title('charge')

        plt.tight_layout()
        plt.savefig(f'{self.molecule.name}_virtual_sites.png')

    def write_xyz(self):
        """
        Write an xyz file containing the atom and virtual site coordinates.
        """

        with open(f'{self.molecule.name}.xyz', 'w+') as xyz_file:
            xyz_file.write(
                f'{len(self.molecule.atoms) + len(self.v_sites_coords)}\n'
                f'xyz file generated with QUBEKit. '
                f'Error with v-site: {min(self.site_errors.values()): .5f} kcal/mol\n'
            )
            for i, atom in enumerate(self.coords):
                xyz_file.write(
                    f'{self.molecule.atoms[i].atomic_symbol}       {atom[0]: .10f}   {atom[1]: .10f}   {atom[2]: .10f}'
                    f'    {self.molecule.atoms[i].partial_charge}\n')

            for site in self.v_sites_coords:
                xyz_file.write(f'X       {site[0][0]: .10f}   {site[0][1]: .10f}   {site[0][2]: .10f}   {site[1]: .10f}\n')
