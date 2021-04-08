#!/usr/bin/env python3

"""
For each atom with fewer than four bonds, generate a list of sample points in a spherical volume around the atoms.
Compare the change in ESP at each of these points for the QM-calculated ESP and the ESP with a v-site.
The v-site can be moved along pre-defined vectors; another v-site can also be added.
If the error is significantly reduced with one or two v-sites, then it is saved for the xml and written to an xyz.
See VirtualSites.fit() for fitting details.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

# DO NOT REMOVE THIS IMPORT. ALTHOUGH IT IS NOT EXPLICITLY CALLED, IT IS NEEDED FOR 3D PLOTTING.
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from qubekit.utils.constants import (
    ANGS_TO_NM,
    BOHR_TO_ANGS,
    ELECTRON_CHARGE,
    J_TO_KCAL_P_MOL,
    M_TO_ANGS,
    PI,
    VACUUM_PERMITTIVITY,
)
from qubekit.utils.helpers import append_to_log

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class VirtualSites:
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
    vdw_radii: Dict[str, float] = {
        "H": 1.44,
        "B": 2.04,
        "C": 1.93,
        "N": 1.83,
        "O": 1.75,
        "F": 1.68,
        "P": 2.07,
        "S": 2.02,
        "Cl": 1.97,
        "Br": 2.10,
        "I": 2.25,
    }

    def __init__(self, molecule: "Ligand", debug: bool = False):
        """
        :param molecule: The usual Ligand molecule object.
        :param debug: Running interactively or not. This will either show an interactive plot of the v-sites,
            or save an image with their final locations.
        """

        self.molecule: "Ligand" = molecule
        self.debug: bool = debug
        self.coords: np.ndarray = self.molecule.coordinates

        # List of tuples where each tuple is the xyz coords of the v-site(s),
        # followed by their charge and index of the parent atom.
        # This list is extended with each atom which has (a) virtual site(s).
        # [((x, y, z), q, atom_index), ... ]
        self.v_sites_coords: List[Tuple[np.ndarray, float, int]] = []

        self.sample_points: Optional[np.ndarray] = None
        self.no_site_esps: Optional[np.ndarray] = None

    @staticmethod
    def spherical_to_cartesian(spherical_coords: np.ndarray) -> np.ndarray:
        """
        :return: Cartesian (x, y, z) coords from the spherical (r, theta, phi) coords.
        """
        r, theta, phi = spherical_coords
        return np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )

    @staticmethod
    def xyz_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        :param point1: coordinates of a point
        :param point2: coordinates of another point
        :return: distance between the two points
        """
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def monopole_esp_one_charge(charge: float, dist: float) -> float:
        """
        Calculate the esp from a monopole at a given distance
        :param charge: charge at atom centre
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: monopole esp value
        """
        return (charge * ELECTRON_CHARGE * ELECTRON_CHARGE) / (
            4 * PI * VACUUM_PERMITTIVITY * dist
        )

    @staticmethod
    def monopole_esp_two_charges(
        charge1: float, charge2: float, dist1: float, dist2: float
    ) -> float:
        """
        Calculate the esp from a monopole with two charges, each a different distance from the point of measurement
        :return: monopole esp value
        """
        return (
            (ELECTRON_CHARGE * ELECTRON_CHARGE) / (4 * PI * VACUUM_PERMITTIVITY)
        ) * (charge1 / dist1 + charge2 / dist2)

    @staticmethod
    def monopole_esp_three_charges(
        charge1: float,
        charge2: float,
        charge3: float,
        dist1: float,
        dist2: float,
        dist3: float,
    ) -> float:
        """
        Calculate the esp from a monopole with three charges, each a different distance from the point of measurement
        :return: monopole esp value
        """
        return (
            (ELECTRON_CHARGE * ELECTRON_CHARGE) / (4 * PI * VACUUM_PERMITTIVITY)
        ) * (charge1 / dist1 + charge2 / dist2 + charge3 / dist3)

    @staticmethod
    def dipole_esp(
        dist_vector: np.ndarray, dipole_moment: np.ndarray, dist: float
    ) -> float:
        """
        Calculate the esp from a dipole at a given sample point.
        :param dist_vector: atom_coords - sample_coords
        :param dipole_moment: dipole moment xyz components from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: dipole esp value
        """
        return (dipole_moment * ELECTRON_CHARGE * ELECTRON_CHARGE).dot(dist_vector) / (
            4 * PI * VACUUM_PERMITTIVITY * dist ** 3
        )

    @staticmethod
    def quadrupole_moment_tensor(
        q_xy: float, q_xz: float, q_yz: float, q_x2_y2: float, q_3z2_r2: float
    ) -> np.ndarray:
        """
        :params: quadrupole moment components from Chargemol output
        :return: quadrupole moment tensor, M
        """
        return np.array(
            [
                [q_x2_y2 / 2 - q_3z2_r2 / 6, q_xy, q_xz],
                [q_xy, -q_x2_y2 / 2 - q_3z2_r2 / 6, q_yz],
                [q_xz, q_yz, q_3z2_r2 / 3],
            ]
        )

    @staticmethod
    def quadrupole_esp(
        dist_vector: np.ndarray, m_tensor: np.ndarray, dist: float
    ) -> float:
        """
        Calculate the esp from a quadrupole at a given distance.
        :param dist_vector: atom_coords - sample_coords
        :param m_tensor: quadrupole moment tensor calculated from Chargemol output
        :param dist: distance from sample_coords to atom_coords
                    (provided as argument to prevent repeated calculation)
        :return: quadrupole esp value
        """
        return (
            3
            * ELECTRON_CHARGE
            * ELECTRON_CHARGE
            * dist_vector.dot(m_tensor * (BOHR_TO_ANGS ** 2)).dot(dist_vector)
        ) / (8 * PI * VACUUM_PERMITTIVITY * dist ** 5)

    @staticmethod
    def cloud_penetration(a: float, b: float, dist: float) -> float:
        """
        Calculate the cloud penetration at a given distance from the atom centre.
        :param a: unitless quantity from DDEC output
        :param b: quantity from DDEC output in units 1/length
        :param dist: distance from sample_coords to atom_coords
        :return: cloud penetration term in SI units
        """
        return (
            (
                ELECTRON_CHARGE
                * ELECTRON_CHARGE
                / (VACUUM_PERMITTIVITY * BOHR_TO_ANGS ** 3)
            )
            * np.exp(a - b * dist)
            * (2 / (b * dist) + 1)
            / (b * b)
        )

    @staticmethod
    def generate_sample_points_relative(vdw_radius: float) -> np.ndarray:
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
        for shell in range(1, shells + 1):
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

        return np.array(relative_sample_points)

    def generate_sample_points_atom(self, atom_index: int) -> np.ndarray:
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

        sample_points = (
            VirtualSites.generate_sample_points_relative(vdw_radius) + atom_coords
        )

        return sample_points

    def generate_esp_atom(self, atom_index: int) -> np.ndarray:
        """
        Using the multipole expansion, calculate the esp at each sample point around an atom.
        :param atom_index: The index of the atom being analysed.
        :return: Ordered list of esp values at each sample point around the atom.
        """

        atom_coords = self.coords[atom_index]

        charge = self.molecule.atoms[atom_index].aim.charge
        dip_data = self.molecule.atoms[atom_index].dipole
        dipole_moment = np.array([dip_data.x, dip_data.y, dip_data.z]) * BOHR_TO_ANGS

        quad_data = self.molecule.atoms[atom_index].quadrupole

        cloud_pen_data = self.molecule.atoms[atom_index].cloud_pen
        a, b = cloud_pen_data.a, cloud_pen_data.b
        b /= BOHR_TO_ANGS

        no_site_esps = []
        for point in self.sample_points:
            dist = VirtualSites.xyz_distance(point, atom_coords)
            dist_vector = point - atom_coords

            mono_esp = VirtualSites.monopole_esp_one_charge(charge, dist)
            dipo_esp = VirtualSites.dipole_esp(dist_vector, dipole_moment, dist)

            m_tensor = VirtualSites.quadrupole_moment_tensor(
                quad_data.q_xy,
                quad_data.q_xz,
                quad_data.q_yz,
                quad_data.q_x2_y2,
                quad_data.q_3z2_r2,
            )
            quad_esp = VirtualSites.quadrupole_esp(dist_vector, m_tensor, dist)

            cloud_pen = VirtualSites.cloud_penetration(a, b, dist)

            v_total = (
                (mono_esp + dipo_esp + quad_esp + cloud_pen)
                * M_TO_ANGS
                * J_TO_KCAL_P_MOL
            )
            no_site_esps.append(v_total)

        return np.array(no_site_esps)

    def generate_atom_mono_esp_two_charges(
        self, atom_index: int, site_charge: float, site_coords: np.ndarray
    ) -> np.ndarray:
        """
        With a virtual site, calculate the monopole esp at each sample point around an atom.
        :param atom_index: The index of the atom being analysed.
        :param site_charge: The charge of the virtual site.
        :param site_coords: numpy array of the xyz position of the virtual site.
        :return: Ordered list of esp values at each sample point around the atom.
        """

        atom_coords = self.coords[atom_index]
        # New charge of the atom, having removed the v-site's charge.
        atom_charge = self.molecule.atoms[atom_index].aim.charge - site_charge

        v_site_esps = []
        for point in self.sample_points:
            dist = VirtualSites.xyz_distance(point, atom_coords)
            site_dist = VirtualSites.xyz_distance(point, site_coords)

            mono_esp = VirtualSites.monopole_esp_two_charges(
                atom_charge, site_charge, dist, site_dist
            )
            v_site_esps.append(mono_esp * M_TO_ANGS * J_TO_KCAL_P_MOL)

        return np.array(v_site_esps)

    def generate_atom_mono_esp_three_charges(
        self,
        atom_index: int,
        q_a: float,
        q_b: float,
        site_a_coords: np.ndarray,
        site_b_coords: np.ndarray,
    ) -> np.ndarray:
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
        # New charge of the atom, having removed the v-sites' charges.
        atom_charge = self.molecule.atoms[atom_index].aim.charge - (q_a + q_b)

        v_site_esps = []
        for point in self.sample_points:
            dist = VirtualSites.xyz_distance(point, atom_coords)
            site_a_dist = VirtualSites.xyz_distance(point, site_a_coords)
            site_b_dist = VirtualSites.xyz_distance(point, site_b_coords)

            mono_esp = VirtualSites.monopole_esp_three_charges(
                atom_charge, q_a, q_b, dist, site_a_dist, site_b_dist
            )
            v_site_esps.append(mono_esp * M_TO_ANGS * J_TO_KCAL_P_MOL)

        return np.array(v_site_esps)

    def get_vector_from_coords(
        self, atom_index: int, n_sites: int = 1, alt: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Given the coords of the atom which will have a v-site and its neighbouring atom(s) coords,
        calculate the vector along which the virtual site will sit.
        :param atom_index: The index of the atom being analysed.
        :param n_sites: The number of virtual sites being placed around the atom.
        :param alt: When placing two sites on an atom with two bonds, there are two placements.
            Is this the usual placement, or the alternative (rotated 90 degrees around the bisecting vector).
        :return Vector(s) along which the v-site will sit. (np array)
            These vectors are scaled dependent on the site's parent atom (see dict below)
        """

        atom = self.molecule.atoms[atom_index]
        atom_coords = self.coords[atom_index]

        # Vary max distance between virtual site and atom coords.
        scale_factor_dict = {
            "H": 1.0,
            "C": 1.0,
            "N": 0.8,
            "O": 1.0,
            "F": 1.0,
            "S": 1.0,
            "Cl": 1.5,
            "Br": 1.5,
            "I": 1.5,
            # May require additional
        }
        scale_factor = scale_factor_dict[atom.atomic_symbol]

        # TODO Differentiate between halogens and carbonyls for 2 site case
        #   (error currently low enough to be irrelevant)
        # e.g. halogens / carbonyls
        if len(atom.bonds) == 1:
            bonded_index = atom.bonds[0]  # [0] is used since bonds is a one item list
            bonded_coords = self.coords[bonded_index]
            r_ab = atom_coords - bonded_coords
            if n_sites == 1:
                return (r_ab / np.linalg.norm(r_ab)) * scale_factor
            return (r_ab / np.linalg.norm(r_ab)) * scale_factor, (
                r_ab / np.linalg.norm(r_ab)
            ) * scale_factor

        # e.g. oxygen
        if len(atom.bonds) == 2:
            bonded_index_b, bonded_index_c = atom.bonds
            bonded_coords_b = self.coords[bonded_index_b]
            bonded_coords_c = self.coords[bonded_index_c]
            r_ab = atom_coords - bonded_coords_b
            r_ac = atom_coords - bonded_coords_c
            if n_sites == 1:
                vec = r_ab + r_ac
                return (vec / np.linalg.norm(vec)) * scale_factor
            vec_a = r_ab + r_ac
            if alt:
                vec_b = np.cross(r_ab, r_ac)
            else:
                vec_b = np.cross((r_ab + r_ac), np.cross(r_ab, r_ac))
            return (vec_a / np.linalg.norm(vec_a)) * scale_factor, (
                vec_b / np.linalg.norm(vec_b)
            ) * scale_factor

        # e.g. nitrogen
        if len(atom.bonds) == 3:
            bonded_index_b, bonded_index_c, bonded_index_d = atom.bonds
            bonded_coords_b = self.coords[bonded_index_b]
            bonded_coords_c = self.coords[bonded_index_c]
            bonded_coords_d = self.coords[bonded_index_d]
            r_vec = np.cross(
                (bonded_coords_b - bonded_coords_c), (bonded_coords_d - bonded_coords_c)
            )
            if n_sites == 1:
                return (r_vec / np.linalg.norm(r_vec)) * scale_factor
            else:
                if atom.atomic_symbol == "N":
                    h_s = []
                    for atom_index in atom.bonds:
                        if self.molecule.atoms[atom_index].atomic_symbol == "H":
                            h_s.append(atom_index)
                    # Special case (amine group); position is slightly different
                    if len(h_s) == 2:
                        h_a_coords = self.coords[h_s[0]]
                        h_b_coords = self.coords[h_s[1]]
                        r_ha = atom_coords - h_a_coords
                        r_hb = atom_coords - h_b_coords

                        return (r_vec / np.linalg.norm(r_vec)) * scale_factor, (
                            (r_ha + r_hb) / np.linalg.norm(r_ha + r_hb)
                        ) * scale_factor
                return (r_vec / np.linalg.norm(r_vec)) * scale_factor, (
                    r_vec / np.linalg.norm(r_vec)
                ) * scale_factor

    def esp_from_lambda_and_charge(
        self, atom_index: int, q: float, lam: float, vec: np.ndarray
    ) -> np.ndarray:
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

    def sites_coords_from_vecs_and_lams(
        self,
        atom_index: int,
        lam_a: float,
        lam_b: float,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def esp_from_lambdas_and_charges(
        self,
        atom_index: int,
        q_a: float,
        q_b: float,
        lam_a: float,
        lam_b: float,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> np.ndarray:
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

        site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(
            atom_index, lam_a, lam_b, vec_a, vec_b
        )

        return self.generate_atom_mono_esp_three_charges(
            atom_index, q_a, q_b, site_a_coords, site_b_coords
        )

    def symm_esp_from_lambdas_and_charges(
        self,
        atom_index: int,
        q: float,
        lam: float,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> np.ndarray:
        """
        Symmetric version of the above. Charges and scale factors are the same for both virtual sites.
        Place v-sites at the correct positions along the vectors by scaling according to the lambdas
        calculate the esp from the atom and the v-sites.
        :param atom_index: The index of the atom being analysed.
        :param q: charge of v-sites a and b
        :param lam: scale factors for vecs a and b
        :param vec_a: vector deciding virtual site position
        :param vec_b: vector deciding virtual site position
        :return: Ordered list of esp values at each sample point
        """

        site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(
            atom_index, lam, lam, vec_a, vec_b
        )

        return self.generate_atom_mono_esp_three_charges(
            atom_index, q, q, site_a_coords, site_b_coords
        )

    def one_site_objective_function(
        self, q_lam: Tuple[float, float], atom_index: int, vec: np.ndarray
    ) -> float:
        """
        Add one site with charge q along vector vec, scaled by lam.
        return the sum of differences at each sample point between the ideal ESP and the calculated ESP.
        """
        site_esps = self.esp_from_lambda_and_charge(atom_index, *q_lam, vec)
        return sum(abs(self.no_site_esps - site_esps))

    def two_sites_objective_function(
        self,
        qa_qb_lama_lamb: Tuple[float, float, float, float],
        atom_index: int,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Add two sites with charges qa, qb along vectors vec_a, vec_b, scaled by lama, lamb.
        return the sum of differences at each sample point between the ideal ESP and the calculated ESP.
        """
        site_esps = self.esp_from_lambdas_and_charges(
            atom_index, *qa_qb_lama_lamb, vec_a, vec_b
        )
        return sum(abs(self.no_site_esps - site_esps))

    def symm_two_sites_objective_function(
        self,
        q_lam: Tuple[float, float],
        atom_index: int,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Add two sites with charge q along vectors vec_a, vec_b scaled by lam.
        This is the symmetric case since the charges and scale factors are the same for each site.
        return the sum of differences at each sample point between the ideal ESP and the calculated ESP.
        """
        site_esps = self.symm_esp_from_lambdas_and_charges(
            atom_index, *q_lam, vec_a, vec_b
        )
        return sum(abs(self.no_site_esps - site_esps))

    @staticmethod
    def two_site_one_bond_constraint_charge(x):
        """
        In the case of a halogen atom being given two virtual sites,
        mimicking the chemistry would imply a positive and negative charge either side of the atom.
        This constraint forces a negative and a positive charge.
        The result of this is usually both charges lying on the C-Halo bond.
        """
        return -1 * x[0] * x[1]

    @staticmethod
    def two_site_one_bond_constraint_lambda(x):
        """
        In the case of a halogen atom being given two virtual sites,
        mimicking the chemistry would imply a positive and negative charge either side of the atom.
        This constraint forces a negative and a positive lambda.
        """
        return -1 * x[2] * x[3]

    @staticmethod
    def two_site_two_bond_constraint(x):
        """
        In the case of two sites and two bonds, the vectors are added or subtracted together,
        rather than one vector per site.
        This constraint ensures the max scaling factor of the vectors combined is less than 1.
        NB They can then be scaled to be up to 1.5 by the scale factor.
        """
        return 1 - x[2] ** 2 - x[3] ** 2

    def fit_one_site(self, atom_index: int):
        """
        Fit method for one site whose parent is <atom_index>
        """
        vec = self.get_vector_from_coords(atom_index, n_sites=1)
        bounds = ((-1.0, 1.0), (-1.0, 1.0))
        one_site_fit = minimize(
            self.one_site_objective_function,
            np.array([0, 1]),
            args=(atom_index, vec),
            bounds=bounds,
        )
        error = one_site_fit.fun / len(self.sample_points)
        q, lam = one_site_fit.x
        one_site_coords = [((vec * lam) + self.coords[atom_index], q, atom_index)]

        return error, one_site_coords

    def fit_two_sites(self, atom_index: int):
        """
        Fit method for two sites whose parent is <atom_index>
        """
        if len(self.molecule.atoms[atom_index].bonds) != 2:
            error, site_coords = self.fit_two_sites_one_or_three_bonds(atom_index)
        else:
            error, site_coords = self.fit_two_sites_two_bonds(atom_index)

        return error, site_coords

    def fit_two_sites_one_or_three_bonds(self, atom_index: int):
        """
        Fit method for two sites whose parent, <atom_index> has one or three bonds
        e.g. uncharged halogens and nitrogens
        """
        vec_a, vec_b = self.get_vector_from_coords(atom_index, n_sites=2)
        bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        if len(self.molecule.atoms[atom_index].bonds) == 1:
            constraint = {
                "type": "ineq",
                "fun": VirtualSites.two_site_one_bond_constraint_charge,
            }
        else:
            constraint = None
        two_site_fit = minimize(
            self.two_sites_objective_function,
            np.array([0.0, 0.0, 1.0, 1.0]),
            args=(atom_index, vec_a, vec_b),
            bounds=bounds,
            constraints=constraint,
        )
        error = two_site_fit.fun / len(self.sample_points)
        q_a, q_b, lam_a, lam_b = two_site_fit.x
        site_a_coords, site_b_coords = self.sites_coords_from_vecs_and_lams(
            atom_index, lam_a, lam_b, vec_a, vec_b
        )
        two_site_coords = [
            (site_a_coords, q_a, atom_index),
            (site_b_coords, q_b, atom_index),
        ]
        return error, two_site_coords

    def fit_two_sites_two_bonds(self, atom_index: int):
        """
        Fit method for two sites whose parent, <atom_index> has two bonds
        e.g. uncharged oxygens
        """
        # Dummy error value to be overwritten
        error = 10000
        two_site_coords = None
        for alt in [True, False]:
            bounds = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
            vec_a, vec_b = self.get_vector_from_coords(atom_index, n_sites=2, alt=alt)
            if self.molecule.enable_symmetry:
                two_site_fit = minimize(
                    self.symm_two_sites_objective_function,
                    np.array([0.0, 1.0]),
                    args=(atom_index, vec_a, vec_b),
                    bounds=bounds[1:3],
                    constraints={
                        "type": "ineq",
                        "fun": VirtualSites.two_site_two_bond_constraint,
                    },
                )
                if (two_site_fit.fun / len(self.sample_points)) < error:
                    error = two_site_fit.fun / len(self.sample_points)
                    q, lam = two_site_fit.x
                    q_a = q_b = q
                    lam_a = lam_b = lam
                    (
                        site_a_coords,
                        site_b_coords,
                    ) = self.sites_coords_from_vecs_and_lams(
                        atom_index, lam_a, lam_b, vec_a, vec_b
                    )
                    two_site_coords = [
                        (site_a_coords, q_a, atom_index),
                        (site_b_coords, q_b, atom_index),
                    ]
            else:
                two_site_fit = minimize(
                    self.two_sites_objective_function,
                    np.array([0.0, 0.0, 1.0, 1.0]),
                    args=(atom_index, vec_a, vec_b),
                    bounds=bounds,
                    constraints={
                        "type": "ineq",
                        "fun": VirtualSites.two_site_two_bond_constraint,
                    },
                )
                if (two_site_fit.fun / len(self.sample_points)) < error:
                    error = two_site_fit.fun / len(self.sample_points)
                    q_a, q_b, lam_a, lam_b = two_site_fit.x
                    (
                        site_a_coords,
                        site_b_coords,
                    ) = self.sites_coords_from_vecs_and_lams(
                        atom_index, lam_a, lam_b, vec_a, vec_b
                    )
                    two_site_coords = [
                        (site_a_coords, q_a, atom_index),
                        (site_b_coords, q_b, atom_index),
                    ]
        return error, two_site_coords

    def fit(self, atom_index: int):
        """
        The error for the objective functions is defined as the sum of differences at each sample point
        between the ideal ESP and the ESP with and without sites.

        * The ESP is first calculated without any virtual sites, if the error is below 1.0, no fitting
        is carried out.
        * Virtual sites are added along pre-defined vectors, and the charges and scale factors of the vectors
        are fit to give the lowest errors.
        * This is done for single sites and two sites (sometimes in two orientations).
        * The two sites may be placed symmetrically, using the bool molecule.symmetry argument.
        * The errors from the sites are printed to terminal, and a plot is produced showing the positions,
        sample points, and charges.
        :param atom_index: The index of the atom being analysed.
        """

        # Calc error in esp when no sites are present
        vec = self.get_vector_from_coords(atom_index, n_sites=1)
        no_site_error = self.one_site_objective_function((0, 1), atom_index, vec)
        no_site_error /= len(self.sample_points)

        # Error in esp is sufficiently low to not require virtual sites.
        if no_site_error <= 1.0:
            return

        one_site_error, one_site_coords = self.fit_one_site(atom_index)

        two_site_error, two_site_coords = self.fit_two_sites(atom_index)

        site_errors = {
            0: no_site_error,
            1: one_site_error,
            2: two_site_error,
        }

        max_err = self.molecule.v_site_error_factor
        if no_site_error < min(one_site_error * max_err, two_site_error * max_err):
            append_to_log(
                self.molecule.home,
                "No virtual site placement has reduced the error significantly.",
                and_print=True,
            )
        elif one_site_error < two_site_error * max_err:
            append_to_log(
                self.molecule.home,
                "The addition of one virtual site was found to be best.",
                and_print=True,
            )
            self.v_sites_coords.extend(one_site_coords)
            self.molecule.NonbondedForce[(atom_index,)].charge -= one_site_coords[0][1]

        else:
            append_to_log(
                self.molecule.home,
                "The addition of two virtual sites was found to be best.",
                and_print=True,
            )
            self.v_sites_coords.extend(two_site_coords)
            self.molecule.NonbondedForce[(atom_index,)].charge -= (
                two_site_coords[0][1] + two_site_coords[1][1]
            )
        append_to_log(
            self.molecule.home,
            f"Errors (kcal/mol):\n"
            f"No Site     One Site     Two Sites\n"
            f"{no_site_error:.4f}      {one_site_error:.4f}       {two_site_error:.4f}",
            and_print=True,
        )
        self.plot(atom_index, site_errors, one_site_coords, two_site_coords)

    def plot(
        self,
        atom_index: int,
        errors: Dict,
        one_site_coords: List[Tuple[np.ndarray, float, float]],
        two_site_coords: List[Tuple[np.ndarray, float, float]],
    ):
        """
        Figure with three subplots.
        All plots show the atoms and bonds as balls and sticks; virtual sites are x's; sample points are dots.
            * Plot showing the positions of the sample points.
            * Plot showing the position of a single virtual site.
            * Plot showing the positions of two virtual sites.
        Errors are included to show the impact of virtual site placements.
        """

        fig = plt.figure(figsize=plt.figaspect(0.33), tight_layout=True)

        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
        cmap = "cool"

        samp_plt = fig.add_subplot(1, 3, 1, projection="3d")
        one_plt = fig.add_subplot(1, 3, 2, projection="3d")
        two_plt = fig.add_subplot(1, 3, 3, projection="3d")

        plots = [samp_plt, one_plt, two_plt]

        # List of tuples where each tuple is the xyz atom coords, followed by their partial charge
        atom_points = [
            (coord, nb_data.charge)  # [((x, y, z), q), ... ]
            for coord, nb_data in zip(self.coords, self.molecule.NonbondedForce)
        ]

        # Add atom positions to all subplots
        for i, plot in enumerate(plots):
            plot.scatter(
                xs=[i[0][0] for i in atom_points],
                ys=[i[0][1] for i in atom_points],
                zs=[i[0][2] for i in atom_points],
                c=[i[1] for i in atom_points],
                marker="o",
                s=200,
                cmap=cmap,
                norm=norm,
            )

            # Plot the bonds as connecting lines
            for bond in self.molecule.bonds:
                plot.plot(
                    xs=[
                        self.coords[bond.atom1_index][0],
                        self.coords[bond.atom2_index][0],
                    ],
                    ys=[
                        self.coords[bond.atom1_index][1],
                        self.coords[bond.atom2_index][1],
                    ],
                    zs=[
                        self.coords[bond.atom1_index][2],
                        self.coords[bond.atom2_index][2],
                    ],
                    c="darkslategrey",
                    alpha=0.5,
                )

        # Left subplot contains the sample point positions
        samp_plt.scatter(
            xs=[i[0] for i in self.sample_points],
            ys=[i[1] for i in self.sample_points],
            zs=[i[2] for i in self.sample_points],
            c="darkslategrey",
            marker="o",
            s=5,
        )
        samp_plt.title.set_text(f"Sample Points Positions\nError: {errors[0]: .5}")

        # Centre subplot contains the single v-site
        one_plt.scatter(
            xs=[i[0][0] for i in one_site_coords],
            ys=[i[0][1] for i in one_site_coords],
            zs=[i[0][2] for i in one_site_coords],
            c=[i[1] for i in one_site_coords],
            marker="x",
            s=200,
            cmap=cmap,
            norm=norm,
        )
        one_plt.title.set_text(f"One Site Position\nError: {errors[1]: .5}")

        # Right subplot contains the two v-sites
        two_plt.scatter(
            xs=[i[0][0] for i in two_site_coords],
            ys=[i[0][1] for i in two_site_coords],
            zs=[i[0][2] for i in two_site_coords],
            c=[i[1] for i in two_site_coords],
            marker="x",
            s=200,
            cmap=cmap,
            norm=norm,
        )
        two_plt.title.set_text(f"Two Sites Positions\nError: {errors[2]: .5}")

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(np.array([]))
        cbar = fig.colorbar(sm)
        cbar.ax.set_title("charge")

        plt.tight_layout()

        if not self.debug:
            atomic_symbol = self.molecule.atoms[atom_index].atomic_symbol
            plt.savefig(
                f"{self.molecule.name}_{atomic_symbol}{atom_index}_virtual_sites.png"
            )

        # Prevent memory leaks
        plt.close()

    def write_xyz(self):
        """
        Write an xyz file containing the atom and virtual site coordinates.
        """

        with open("xyz_with_extra_point_charges.xyz", "w+") as xyz_file:
            xyz_file.write(
                f"{len(self.molecule.atoms) + len(self.v_sites_coords)}\n"
                f"xyz file generated with QUBEKit.\n"
            )
            for atom_index, atom in enumerate(self.coords):
                xyz_file.write(
                    f"{self.molecule.atoms[atom_index].atomic_symbol}       {atom[0]: .10f}   {atom[1]: .10f}   {atom[2]: .10f}"
                    f"   {self.molecule.NonbondedForce[(atom_index,)].charge: .6f}\n"
                )

                for site in self.v_sites_coords:
                    if site[2] == atom_index:
                        xyz_file.write(
                            f"X       {site[0][0]: .10f}   {site[0][1]: .10f}   {site[0][2]: .10f}   {site[1]: .6f}\n"
                        )

    def save_virtual_sites(self):
        """
        Take the v_site_coords generated and insert them into the Ligand object as molecule.extra_sites.

        Uses the coordinates to generate the necessary position vectors to be used in the xml.
        """

        topology = self.molecule.to_topology()
        for site_number, site in enumerate(self.v_sites_coords):

            site_data = {}

            site_coords, site_charge, parent = site
            site_data["parent_index"] = parent
            site_data["charge"] = site_charge

            closest_atoms = list(topology.neighbors(parent))
            if (len(closest_atoms) < 2) or (len(self.molecule.atoms[parent].bonds) > 3):
                for atom in list(topology.neighbors(closest_atoms[0])):
                    if atom not in closest_atoms and atom != parent:
                        closest_atoms.append(atom)
                        break

            # Get the xyz coordinates of the reference atoms
            parent_coords = self.coords[parent]
            close_a_coords = self.coords[closest_atoms[0]]
            close_b_coords = self.coords[closest_atoms[1]]

            site_data["closest_a_index"] = closest_atoms[0]
            site_data["closest_b_index"] = closest_atoms[1]

            parent_atom = self.molecule.atoms[parent]
            if parent_atom.atomic_symbol == "N" and len(parent_atom.bonds) == 3:
                close_c_coords = self.coords[closest_atoms[2]]
                site_data["closest_c_index"] = closest_atoms[2]

                x_dir = (
                    (close_a_coords + close_b_coords + close_c_coords) / 3
                ) - parent_coords
                x_dir /= np.linalg.norm(x_dir)

                site_data["p2"] = 0
                site_data["p3"] = 0

                site_data["o_weights"] = [1.0, 0.0, 0.0, 0.0]  # SUM MUST BE 1.0
                site_data["x_weights"] = [-1.0, 0.33333333, 0.33333333, 0.33333333]
                site_data["y_weights"] = [1.0, -1.0, 0.0, 0.0]

            else:
                x_dir = close_a_coords - parent_coords
                x_dir /= np.linalg.norm(x_dir)

                z_dir = np.cross(
                    (close_a_coords - parent_coords), (close_b_coords - parent_coords)
                )
                z_dir /= np.linalg.norm(z_dir)

                y_dir = np.cross(z_dir, x_dir)

                p2 = float(
                    np.dot((site_coords - parent_coords), y_dir.reshape(3, 1))
                    * ANGS_TO_NM
                )
                site_data["p2"] = round(p2, 4)
                p3 = float(
                    np.dot((site_coords - parent_coords), z_dir.reshape(3, 1))
                    * ANGS_TO_NM
                )
                site_data["p3"] = round(p3, 4)

                site_data["o_weights"] = [1.0, 0.0, 0.0]  # SUM MUST BE 1.0
                site_data["x_weights"] = [-1.0, 1.0, 0.0]
                site_data["y_weights"] = [-1.0, 0.0, 1.0]

            # Get the local coordinate positions
            p1 = float(
                np.dot((site_coords - parent_coords), x_dir.reshape(3, 1)) * ANGS_TO_NM
            )
            site_data["p1"] = round(p1, 4)

            self.molecule.extra_sites.create_site(**site_data)

    def calculate_virtual_sites(self):
        """
        Main worker method.
        Loop over all atoms in the molecule and decide which may need v-sites.
        Fit the ESP accordingly and store v-sites if they improve error.
        If any v-sites are found to be useful, write them to an xyz and store them in the Ligand object
        """

        for atom_index, atom in enumerate(self.molecule.atoms):
            if len(atom.bonds) < 4:
                self.sample_points = self.generate_sample_points_atom(atom_index)
                self.no_site_esps = self.generate_esp_atom(atom_index)
                self.fit(atom_index)

        if self.v_sites_coords:
            self.save_virtual_sites()

        if not self.debug:
            self.write_xyz()
