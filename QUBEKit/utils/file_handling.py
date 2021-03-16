"""
Purpose of this file is to read various inputs and produce the info required for
    Ligand() or Protein()
Should be very little calculation here, simply file reading and some small validations / checks
"""
import os

import numpy as np

from QUBEKit.molecules import ExtraSite
from QUBEKit.utils.constants import ANGS_TO_NM
from QUBEKit.utils.datastructures import CustomNamespace


class ExtractChargeData:
    """
    Choose between extracting (the more extensive) data from Chargemol, or from ONETEP.
    Symmetrise if desired (config option)
    Store all info back into the molecule object; ensure ddec data and atom partial charges match
    """

    def __init__(self, molecule):
        self.molecule = molecule

    def extract_charge_data(self):
        if self.molecule.charges_engine.casefold() == "chargemol":
            self._extract_charge_data_chargemol()
        elif self.molecule.charges_engine.casefold() == "onetep":
            self._extract_charge_data_onetep()
        else:
            raise NotImplementedError(
                "Currently, the only valid charge engines in QUBEKit are ONETEP and Chargemol."
            )

        if self.molecule.enable_symmetry:
            self._apply_symmetrisation()

        # Ensure the partial charges in the atom container are also changed.
        for molecule_atom, ddec_atom in zip(
            self.molecule.atoms, self.molecule.ddec_data.values()
        ):
            molecule_atom.partial_charge = ddec_atom.charge

    def _extract_charge_data_chargemol(self):
        """
        From Chargemol output files, extract the necessary parameters for calculation of L-J.

        :returns: 3 CustomNamespaces, ddec_data; dipole_moment_data; and quadrupole_moment_data
        ddec_data used for calculating monopole esp and L-J values (used by both LennardJones and Charges classes)
        dipole_moment_data used for calculating dipole esp
        quadrupole_moment_data used for calculating quadrupole esp
        """

        if self.molecule.ddec_version == 6:
            net_charge_file_name = "DDEC6_even_tempered_net_atomic_charges.xyz"

        elif self.molecule.ddec_version == 3:
            net_charge_file_name = "DDEC3_net_atomic_charges.xyz"

        else:
            raise ValueError("Unsupported DDEC version; please use version 3 or 6.")

        try:
            with open(net_charge_file_name, "r+") as charge_file:
                lines = charge_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(
                "Cannot find the DDEC output file.\nThis could be indicative of several issues.\n"
                "Please check Chargemol is installed in the correct location and that the configs"
                " point to that location."
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
                f"Cannot find charge or cloud penetration data in {net_charge_file_name}."
            )

        ddec_data = {}
        dipole_moment_data = {}
        quadrupole_moment_data = {}

        cloud_pen_data = {}

        for line in lines[ddec_start_pos : ddec_start_pos + atom_total]:
            # _'s are the xyz coords, then the quadrupole moment tensor eigenvalues.
            (
                atom_count,
                atomic_symbol,
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
            ddec_data[atom_index] = CustomNamespace(
                atomic_symbol=atomic_symbol,
                charge=float(charge),
                volume=None,
                r_aim=None,
                b_i=None,
                a_i=None,
            )

            dipole_moment_data[atom_index] = CustomNamespace(
                x_dipole=float(x_dipole),
                y_dipole=float(y_dipole),
                z_dipole=float(z_dipole),
            )

            quadrupole_moment_data[atom_index] = CustomNamespace(
                q_xy=float(q_xy),
                q_xz=float(q_xz),
                q_yz=float(q_yz),
                q_x2_y2=float(q_x2_y2),
                q_3z2_r2=float(q_3z2_r2),
            )

        for line in lines[cloud_pen_pos : cloud_pen_pos + atom_total]:
            # _'s are the xyz coords and the r_squared.
            atom_count, atomic_symbol, _, _, _, a, b, _ = line.split()
            atom_index = int(atom_count) - 1
            cloud_pen_data[atom_index] = CustomNamespace(
                atomic_symbol=atomic_symbol, a=float(a), b=float(b)
            )

        r_cubed_file_name = "DDEC_atomic_Rcubed_moments.xyz"

        with open(r_cubed_file_name, "r+") as vol_file:
            lines = vol_file.readlines()

        vols = [float(line.split()[-1]) for line in lines[2 : atom_total + 2]]

        for atom_index in ddec_data:
            ddec_data[atom_index].volume = vols[atom_index]

        self.molecule.ddec_data = ddec_data
        self.molecule.dipole_moment_data = dipole_moment_data
        self.molecule.quadrupole_moment_data = quadrupole_moment_data
        self.molecule.cloud_pen_data = cloud_pen_data

    def _extract_charge_data_onetep(self):
        """
        From ONETEP output files, extract the necessary parameters for calculation of L-J.
        Insert data into ddec_data in standard format.
        Used exclusively by LennardJones class.
        """

        # Just fill in None values until they are known
        ddec_data = {
            i: CustomNamespace(
                atomic_symbol=atom.atomic_symbol,
                charge=None,
                volume=None,
                r_aim=None,
                b_i=None,
                a_i=None,
            )
            for i, atom in enumerate(self.molecule.atoms)
        }

        # Second file contains the rest (charges, dipoles and volumes):
        ddec_output_file = (
            "ddec.onetep" if os.path.exists("ddec.onetep") else "iter_1/ddec.onetep"
        )
        with open(ddec_output_file, "r") as file:
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
            for line in lines[charge_pos : charge_pos + len(self.molecule.atoms)]
        ]

        # Add the AIM-Valence and the AIM-Core to get V^AIM
        volumes = [
            float(line.split()[2]) + float(line.split()[3])
            for line in lines[vol_pos : vol_pos + len(self.molecule.atoms)]
        ]

        for atom_index in ddec_data:
            ddec_data[atom_index].charge = charges[atom_index]
            ddec_data[atom_index].volume = volumes[atom_index]

        self.molecule.ddec_data = ddec_data

    def _apply_symmetrisation(self):
        """
        Using the atoms picked out to be symmetrised:
        apply the symmetry to the charge and volume values.
        Mutates the non_bonded_force dict
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


def extract_extra_sites_onetep(molecule):
    """
    Gather the extra sites from the xyz file and insert them into the molecule object.
    * Find parent and 2 reference atoms
    * Calculate the local coords site
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


def make_and_change_into(name):
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
