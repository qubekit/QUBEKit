#!/usr/bin/env python3

"""
Purpose of this file is to read various inputs and produce the info required for
    Ligand() or Protein()
Should be very little calculation here, simply file reading and some small validations / checks
"""

import os
from typing import List
from typing_extensions import Literal

import numpy as np

from QUBEKit.molecules import CloudPen, Dipole, ExtraSite, Ligand, Quadrupole
from QUBEKit.utils.constants import ANGS_TO_NM


class ExtractChargeData:
    """
    Choose between extracting (the more extensive) data from Chargemol, or from ONETEP.
    Store all info back into the molecule object
    """

    @classmethod
    def read_files_chargemol(
        cls,
        molecule: Ligand,
        dir_path: str,
        ddec_version: Literal[3, 6],
    ) -> Ligand:
        updated_mol = cls._extract_charge_data_chargemol(
            molecule, dir_path, ddec_version
        )
        return updated_mol

    @classmethod
    def read_files_onetep(cls, molecule: Ligand, dir_path: str) -> Ligand:
        updated_mol = cls._extract_charge_data_onetep(molecule, dir_path)
        return updated_mol

    @classmethod
    def _extract_charge_data_chargemol(
        cls,
        molecule: Ligand,
        dir_path: str,
        ddec_version: Literal[3, 6],
    ):
        """
        From Chargemol output files, extract the necessary parameters for calculation of L-J.
        Args:
            molecule: Usual molecule object.
            ddec_version: Which ddec version was used to run chargemol? 3 or 6.
            dir_path: Directory containing the ddec output files.
        Return:
            Updated molecule object.
        """

        if ddec_version == 6:
            net_charge_file_path = os.path.join(
                dir_path, "DDEC6_even_tempered_net_atomic_charges.xyz"
            )

        elif ddec_version == 3:
            net_charge_file_path = os.path.join(
                dir_path, "DDEC3_net_atomic_charges.xyz"
            )

        else:
            raise ValueError("Unsupported DDEC version; please use version 3 or 6.")

        try:
            with open(net_charge_file_path, "r+") as charge_file:
                lines = charge_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Cannot find the DDEC output file.\nThis could be indicative of several issues.\n"
                "Please check Chargemol is installed in the correct location and that the configs "
                "point to that location."
            )

        # Find number of atoms
        atom_total = int(lines[0])

        # Find data markers:
        ddec_start_pos, cloud_pen_pos = 0, 0
        for pos, row in enumerate(lines):
            if "The following XYZ" in row:
                ddec_start_pos = pos + 2

            # [sic]
            elif "The sperically averaged" in row:
                cloud_pen_pos = pos + 2

            if ddec_start_pos and cloud_pen_pos:
                break
        else:
            raise EOFError(
                f"Cannot find charge or cloud penetration data in {net_charge_file_path}."
            )

        for line in lines[ddec_start_pos : ddec_start_pos + atom_total]:
            # _'s are the atomic symbol, xyz coords, then the quadrupole moment tensor eigenvalues.
            (
                atom_count,
                _,
                _,
                _,
                _,
                charge,
                x_dipole,
                y_dipole,
                z_dipole,
                _,
                q_xy,
                q_xz,
                q_yz,
                q_x2_y2,
                q_3z2_r2,
                *_,
            ) = line.split()

            # File counts from 1 not 0; thereby requiring -1 to get the index.
            atom_index = int(atom_count) - 1

            molecule.atoms[atom_index].aim.charge = float(charge)

            molecule.atoms[atom_index].dipole = Dipole(
                x=float(x_dipole),
                y=float(y_dipole),
                z=float(z_dipole),
            )

            molecule.atoms[atom_index].quadrupole = Quadrupole(
                q_xy=float(q_xy),
                q_xz=float(q_xz),
                q_yz=float(q_yz),
                q_x2_y2=float(q_x2_y2),
                q_3z2_r2=float(q_3z2_r2),
            )

        for line in lines[cloud_pen_pos : cloud_pen_pos + atom_total]:
            # _'s are the xyz coords and the r_squared.
            atom_count, _, _, _, _, a, b, _ = line.split()
            atom_index = int(atom_count) - 1
            molecule.atoms[atom_index].cloud_pen = CloudPen(
                a=float(a),
                b=float(b),
            )

        r_cubed_file_name = os.path.join(dir_path, "DDEC_atomic_Rcubed_moments.xyz")

        with open(r_cubed_file_name, "r+") as vol_file:
            lines = vol_file.readlines()

        volumes = [float(line.split()[-1]) for line in lines[2 : atom_total + 2]]

        for atom_index in range(atom_total):
            molecule.atoms[atom_index].aim.vol = volumes[atom_index]

        return molecule

    @classmethod
    def _extract_charge_data_onetep(cls, molecule: Ligand, dir_path: str):
        """
        From ONETEP output files, extract the necessary parameters for calculation of L-J.
        Args:
            molecule: Usual molecule object.
            dir_path: Directory containing the ddec output files.
        Return:
            Updated molecule object.
        """

        # Second file contains the rest (charges, dipoles and volumes):
        ddec_output_file = (
            "ddec.onetep" if os.path.exists("ddec.onetep") else "iter_1/ddec.onetep"
        )

        ddec_file_path = os.path.join(dir_path, ddec_output_file)

        with open(ddec_file_path, "r") as file:
            lines = file.readlines()

        charge_pos, vol_pos = None, None
        for pos, line in enumerate(lines):

            # Charges marker in file:
            if "DDEC density" in line:
                charge_pos = pos + 7

            # Volumes marker in file:
            if "DDEC Radial" in line:
                vol_pos = pos + 4

        if any(position is None for position in [charge_pos, vol_pos]):
            raise EOFError(
                "Cannot locate charges and / or volumes in ddec.onetep file."
            )

        charges = [
            float(line.split()[-1])
            for line in lines[charge_pos : charge_pos + molecule.n_atoms]
        ]

        # Add the AIM-Valence and the AIM-Core to get V^AIM
        volumes = [
            float(line.split()[2]) + float(line.split()[3])
            for line in lines[vol_pos : vol_pos + molecule.n_atoms]
        ]

        for atom_index in range(molecule.n_atoms):
            molecule.atoms[atom_index].aim.vol = volumes[atom_index]
            molecule.atoms[atom_index].aim.charge = charges[atom_index]

        return molecule

    def _apply_symmetrisation(self):
        """
        TODO FIX and move. Don't use to symmetrise the raw params.
        Using the atoms picked out to be symmetrised:
        apply the symmetry to the charge and volume values.
        Mutates the non_bonded_force dict.
        """

        atom_types = {}
        for key, val in self.molecule.atom_types.items():
            atom_types.setdefault(val, []).append(key)

        # Find the average charge / volume values for each sym_set.
        # A sym_set is atoms which should have the same charge / volume values (e.g. methyl H's).
        for sym_set in atom_types.values():
            charge = sum(
                self.molecule.ddec_data[atom].charge for atom in sym_set
            ) / len(sym_set)
            volume = sum(
                self.molecule.ddec_data[atom].volume for atom in sym_set
            ) / len(sym_set)

            # Store the new values.
            for atom in sym_set:
                self.molecule.ddec_data[atom].charge = round(charge, 6)
                self.molecule.ddec_data[atom].volume = round(volume, 6)


def extract_c8_params(dir_path: str) -> List[float]:
    """
    Extract the C8 dispersion coefficients from the MCLF calculation's output file.
    Args:
        dir_path: directory containing the C8 xyz file from Chargemol.
    returns:
        ordered list of the c8 params for each atom in molecule.
    """

    with open(os.path.join(dir_path, "MCLF_C8_dispersion_coefficients.xyz")) as c8_file:
        lines = c8_file.readlines()
        for i, line in enumerate(lines):
            if line.startswith(" The following "):
                lines = lines[i + 2 : -2]
                break
        else:
            raise EOFError("Cannot locate c8 parameters in file.")

        # c8 params IN ATOMIC UNITS
        c8_params = [float(line.split()[-1].strip()) for line in lines]

    return c8_params


def extract_extra_sites_onetep(molecule: Ligand):
    """
    Gather the extra sites from the xyz file and insert them into the molecule object.
    * Find parent and 2 reference atoms
    * Calculate the local coords site
    Args:
        molecule: Usual molecule object.
    """

    with open("xyz_with_extra_point_charges.xyz") as xyz_sites:
        lines = xyz_sites.readlines()

    extra_sites = dict()
    parent = 0
    site_number = 0

    for i, line in enumerate(lines[2:]):
        if line.split()[0] != "X":
            parent += 1
            # Search the following entries for sites connected to this atom
            for virtual_site in lines[i + 3 :]:
                site_data = ExtraSite()
                element, *site_coords, site_charge = virtual_site.split()
                # Not a virtual site:
                if element != "X":
                    break
                else:
                    site_coords = np.array([float(coord) for coord in site_coords])

                    closest_atoms = list(molecule.topology.neighbors(parent))
                    if (len(closest_atoms) < 2) or (
                        len(molecule.atoms[parent].bonds) > 3
                    ):
                        for atom in list(molecule.topology.neighbors(closest_atoms[0])):
                            if atom not in closest_atoms and atom != parent:
                                closest_atoms.append(atom)
                                break

                    # Get the xyz coordinates of the reference atoms
                    coords = (
                        molecule.coords["qm"]
                        if molecule.coords["qm"] is not []
                        else molecule.coords["input"]
                    )
                    parent_coords = coords[parent]
                    close_a_coords = coords[closest_atoms[0]]
                    close_b_coords = coords[closest_atoms[1]]

                    site_data.parent_index = parent
                    site_data.closest_a_index = closest_atoms[0]
                    site_data.closest_b_index = closest_atoms[1]

                    parent_atom = molecule.atoms[parent]
                    if parent_atom.atomic_symbol == "N" and len(parent_atom.bonds) == 3:
                        close_c_coords = coords[closest_atoms[2]]
                        site_data.closest_c_index = closest_atoms[2]

                        x_dir = (
                            (close_a_coords + close_b_coords + close_c_coords) / 3
                        ) - parent_coords
                        x_dir /= np.linalg.norm(x_dir)

                        site_data.p2 = 0
                        site_data.p3 = 0

                        site_data.o_weights = [1.0, 0.0, 0.0, 0.0]
                        site_data.x_weights = [-1.0, 0.33333333, 0.33333333, 0.33333333]
                        site_data.y_weights = [1.0, -1.0, 0.0, 0.0]

                    else:
                        x_dir = close_a_coords - parent_coords
                        x_dir /= np.linalg.norm(x_dir)

                        z_dir = np.cross(
                            (close_a_coords - parent_coords),
                            (close_b_coords - parent_coords),
                        )
                        z_dir /= np.linalg.norm(z_dir)

                        y_dir = np.cross(z_dir, x_dir)

                        p2 = float(
                            np.dot((site_coords - parent_coords), y_dir.reshape(3, 1))
                            * ANGS_TO_NM
                        )
                        site_data.p2 = round(p2, 4)
                        p3 = float(
                            np.dot((site_coords - parent_coords), z_dir.reshape(3, 1))
                            * ANGS_TO_NM
                        )
                        site_data.p3 = round(p3, 4)

                        site_data.o_weights = [1.0, 0.0, 0.0]
                        site_data.x_weights = [-1.0, 1.0, 0.0]
                        site_data.y_weights = [-1.0, 0.0, 1.0]

                    p1 = float(
                        np.dot((site_coords - parent_coords), x_dir.reshape(3, 1))
                        * ANGS_TO_NM
                    )
                    site_data.p1 = round(p1, 4)

                    extra_sites[site_number] = site_data

                    site_number += 1

    molecule.extra_sites = extra_sites


def make_and_change_into(name: str):
    """
    - Attempt to make a directory with name <name>, don't fail if it exists.
    - Change into the directory.
    """

    try:
        os.mkdir(name)
    except FileExistsError:
        pass
    finally:
        os.chdir(name)


def get_data(relative_path: str) -> str:
    """
    Get the file path to some data in the qubekit package.
    Parameters
    ----------
    relative_path
        The relative path to the file that should be loaded from QUBEKit/data.
    Returns
    -------
    str
        The absolute path to the requested file.
    """
    from pkg_resources import resource_filename

    fn = resource_filename("QUBEKit", os.path.join("data", relative_path))
    if not os.path.exists(fn):
        raise ValueError(
            f"{relative_path} does not exist. If you have just added it, you'll have to re-install"
        )

    return fn
