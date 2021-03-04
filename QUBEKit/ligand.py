#!/usr/bin/env python3

"""
TODO ligand.py Refactor:
    DO:
        Move module-specific methods such as openmm_coordinates(); read_tdrive(); read_geometric_traj
            to their relevant files/classes
        Fix naming; consistency wrt get/find; clarity on all of the dihedral variables
            (what is dih_start, how is it different to di_starts etc)
        Perform checks after reading input (check_names_are_unique(), validate_info(), etc)
    CONSIDER:
        Add typing; especially for class variables
            Careful wrt complex variables such as coords, atoms, etc
        Remove / replace DefaultsMixin with inheritance, dict or some other solution
        Remove any repeated or unnecessary variables
            Should state be handled in ligand or run?
            Should testing be handled in ligand or tests?
        Change the structure and type of some variables for clarity
            Should coords actually be a class instead of a dict?
            Do we access via index too often; should we use e.g. SimpleNamespaces/NamedTupleS?
        Split ligand.py into a package with three+ files:
            base/defaults/configs.py; ligand.py; protein.py (where should Atom() go?)
        Be more strict about public/private class/method/function naming?
"""

import os
import pickle
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from xml.dom.minidom import parseString

import networkx as nx
import numpy as np

from QUBEKit.engines import RDKit
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import (
    Atom,
    ExtraSite,
    TorsionDriveData,
    ParameterTags,
)
from QUBEKit.utils.exceptions import FileTypeError, MissingReferenceData
from QUBEKit.utils.file_handling import ReadInput, ReadInputProtein


class DefaultsMixin:
    """
    This class holds all of the default configs from the config file.
    It's effectively a placeholder for all of the attributes which may
    be changed by editing the config file(s).

    See the config class for details of all these params.

    It's a mixin because:
        * Normal multiple inheritance doesn't make sense in this context
        * Composition would be a bit messier and may require stuff like:
            mol = Ligand('methane.pdb', 'methane')
            mol.defaults.threads
            >> 2

            rather than the nice clean:
            mol.threads
            >> 2
        * Mixin is cleaner and clearer with respect to super() calls.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.theory = "wB97XD"
        self.basis = "6-311++G(d,p)"
        self.vib_scaling = 1
        self.threads = 4
        self.memory = 4
        self.convergence = "GAU_TIGHT"
        self.iterations = 350
        self.bonds_engine = "g09"
        self.density_engine = "g09"
        self.charges_engine = "chargemol"
        self.ddec_version = 6
        self.dielectric = 4.0
        self.geometric = True
        self.solvent = True
        self.enable_symmetry = True
        self.enable_virtual_sites = True
        self.v_site_error_factor = 1.005

        self.dih_start = -165
        self.increment = 15
        self.dih_end = 180
        self.t_weight = "infinity"
        self.opt_method = "BFGS"
        self.refinement_method = "SP"
        self.tor_limit = 20
        self.div_index = 0
        self.parameter_engine = "antechamber"
        self.l_pen = 0.0
        self.mm_opt_method = "openmm"
        self.relative_to_global = False

        self.excited_state = False
        self.excited_theory = "TDA"
        self.n_states = 3
        self.excited_root = 1
        self.use_pseudo = False
        self.pseudo_potential_block = ""

        self.chargemol = "/home/<QUBEKit_user>/chargemol_09_26_2017"
        self.log = 999
        # Internal for QUBEKit testing; stops decorators from trying to log to a file unnecessarily.
        self.testing = False


class Molecule:
    """Base class for ligands and proteins."""

    def __init__(self, mol_input, name=None):
        """
        # Namings
        name                    str; Molecule name e.g. 'methane'
        is_protein              bool; whether or not the molecule is a protein

        # Structure
        coords                  Dict of numpy arrays of the coords where the keys are the input type ('mm', 'qm', etc)
        topology                networkx Graph() object. Contains connection information for molecule
        symm_hs
        qm_energy

        # XML Info
        xml_tree                An XML class object containing the force field values
        AtomTypes               dict of lists; basic non-symmetrised atoms types for each atom in the molecule
                                e.g. {0, ['C1', 'opls_800', 'C800'], 1: ['H1', 'opls_801', 'H801'], ... }
        extra_sites
        qm_scans                Dictionary of central scanned bonds and there energies and structures

        # This section has different units due to it interacting with OpenMM
        HarmonicBondForce       Dictionary of equilibrium distances and force constants stored under the bond tuple.
                                {(1, 2): [0.108, 405.65]} (nano meters, kJ/mol)
        HarmonicAngleForce      Dictionary of equilibrium angles and force constants stored under the angle tuple
                                e.g. {(2, 1, 3): [2.094395, 150.00]} (radians, kJ/mol)
        PeriodicTorsionForce    Dictionary of lists of the torsions values [periodicity, k, phase] stored under the
                                dihedral tuple with an improper tag only for improper torsions
                                e.g. {(3, 1, 2, 6): [[1, 0.6, 0], [2, 0, 3.141592653589793], ... Improper]}
        NonbondedForce          OrderedDict; L-J params. Keys are atom index, vals are [charge, sigma, epsilon]


        dih_start
        dih_end
        increments

        combination             str; Combination rules e.g. 'opls'

        # QUBEKit Internals
        state                   str; Describes the stage the analysis is in for pickling and unpickling
        config_file             str or path; the config file used for the execution
        restart                 bool; is the current execution starting from the beginning (False) or restarting (True)?
        """

        self.mol_input: Union[ReadInput, str] = mol_input
        self.name: str = name
        self.is_protein: bool = False

        self.rdkit_mol = None

        # Structure
        self.coords: Dict = {"input": [], "mm": [], "qm": [], "temp": [], "traj": []}
        self.topology = None
        self.atoms: Optional[List[Atom]] = None
        self.symm_hs = None
        self.qm_energy: Optional[float] = None
        self.charge: int = 0
        self.multiplicity: int = 1
        self.qm_scans = None
        self.qm_energy = None
        self.charge = 0
        self.multiplicity = 1
        self.qm_scans: Optional[List[TorsionDriveData]] = None
        self.scan_order = None
        self.descriptors = None

        # XML Info
        self.AtomTypes: Optional[Dict] = None
        self.extra_sites: Optional[Dict[int, ExtraSite]] = None
        self.HarmonicBondForce: Optional[Dict] = None
        self.HarmonicAngleForce: Optional[Dict] = None
        self.PeriodicTorsionForce: Optional[Dict] = None
        self.NonbondedForce: Optional[Dict] = None

        # Dihedral settings
        self.dih_starts: Dict = {}
        self.dih_ends: Dict = {}
        self.increments: Dict = {}
        # this holds the groups which should not be considered rotatable
        self.methyl_amine_nitride_cores: Optional[Dict] = None

        self.combination: str = "amber"

        # QUBEKit internals
        self.state: Optional[str] = None
        self.config_file: str = "master_config.ini"
        self.restart: bool = False
        # self.atom_types = None
        self.verbose: bool = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def __str__(self, trunc=False):
        """
        Prints the Molecule class objects' names and values one after another with new lines between each.
        Mostly just used for logging, debugging and displaying the results at the end of a run.
        If trunc is set to True:
            Check the items being printed:
                If they are empty or None -> skip over them
                If they're short (<120 chars) -> print them as normal
                Otherwise -> print a truncated version of them.
        If trunc is set to False:
            Just print everything (all key: value pairs) as is with a little extra spacing.
        """

        return_str = ""

        if trunc:
            for key, val in self.__dict__.items():

                # Just checking (if val) won't work as truth table is ambiguous for length > 1 arrays
                # I know this is gross, but it's the best of a bad situation.
                try:
                    bool(val)
                # Catch numpy array truth table error
                except ValueError:
                    continue

                # Ignore NoneTypes and empty lists / dicts etc unless type is int (charge = 0 for example)
                if val is not None and (val or isinstance(val, int)):
                    return_str += f"\n{key} = "

                    # if it's smaller than 120 chars: print it as is. Otherwise print a version cut off with "...".
                    if len(str(key) + str(val)) < 120:
                        # Print the repr() not the str(). This means generator expressions etc appear too.
                        return_str += repr(val)
                    else:
                        return_str += repr(val)[: 121 - len(str(key))] + "..."

        else:
            for key, val in self.__dict__.items():
                # Return all objects as {ligand object name} = {ligand object value(s)} without any special formatting.
                return_str += f"\n{key} = {repr(val)}\n"

        return return_str

    # def check_names_are_unique(self):
    #     """
    #     To prevent problems occurring with some atoms perceived to be the same,
    #     check the atom names to ensure they are all unique.
    #     If some are the same, reset all atom names to be: f'{atomic_symbol}{index}'.
    #     This ensure they are all unique.
    #     """
    #
    #     atom_names = [atom.atom_name for atom in self.atoms]
    #     # If some atom names aren't unique
    #     if len(set(atom_names)) < len(atom_names):
    #         # Change the atom name only; everything else is the same as it was.
    #         self.atoms = [
    #             Atom(
    #                 atomic_number=atom.atomic_number,
    #                 atom_index=atom.atom_index,
    #                 atom_name=f'{atom.atomic_symbol}{i}',
    #                 partial_charge=atom.partial_charge,
    #                 formal_charge=atom.formal_charge
    #             )
    #             for i, atom in enumerate(self.atoms)
    #         ]
    #
    # def _validate_info(self, topology, atoms, coords, input_type, rdkit_molecule=None, descriptors=None):
    #     """
    #     Check if the provided information should be stored or not
    #     :param topology: networkx graph of the topology
    #     :param atoms: a list of Atom objects
    #     :param coords: a numpy array of the coords
    #     :param rdkit_molecule: the rdkit molecule we have extracted the info from
    #     :param descriptors: a dictionary of the rdkit descriptors
    #     :return: the updated ligand object
    #     """
    #
    #     # Now check we instancing the ligand if we are then store the info
    #     if input_type == 'input':
    #         self.topology = topology
    #         self.atoms = atoms
    #         self.descriptors = descriptors
    #         self.coords[input_type] = coords
    #         self.rdkit_mol = rdkit_molecule
    #     else:
    #         # Check if the new topology is the same then store the new coordinates
    #         if nx.algorithms.is_isomorphic(self.topology, topology):
    #             self.coords[input_type] = coords
    #         else:
    #             raise TopologyMismatch('Topologies are not the same; cannot store coordinates.')

    def add_qm_scan(self, scan_data: TorsionDriveData) -> None:
        """
        Save the torsion drive data into the ligand object.
        """
        if scan_data.__class__ != TorsionDriveData:
            raise MissingReferenceData(
                f"The refernce data must be in the form of the torsion drive data class."
            )
        else:
            if self.qm_scans is None:
                self.qm_scans = []
            self.qm_scans.append(scan_data)

    def get_atom_with_name(self, name):
        """
        Search through the molecule for an atom with that name and return it when found
        :param name: The name of the atom we are looking for
        :return: The QUBE Atom object with the name
        """

        for atom in self.atoms:
            if atom.atom_name == name:
                return atom
        raise AttributeError("No atom found with that name.")

    def read_geometric_traj(self, trajectory):
        """
        Read in the molecule coordinates to the traj holder from a geometric optimisation using qcengine.
        :param trajectory: The qcengine trajectory

        TODO Move to QCEngine()
        """

        for frame in trajectory:
            opt_traj = []
            # Convert coordinates from bohr to angstroms
            geometry = np.array(frame["molecule"]["geometry"]) * constants.BOHR_TO_ANGS
            for i, atom in enumerate(frame["molecule"]["symbols"]):
                opt_traj.append(
                    [geometry[0 + i * 3], geometry[1 + i * 3], geometry[2 + i * 3]]
                )
            self.coords["traj"].append(np.array(opt_traj))

    @property
    def has_unique_atom_names(self) -> bool:
        """
        Check if the molecule has unique atom names or not this will help with pdb file writing.
        """
        atom_names = set([atom.atom_name for atom in self.atoms])
        if len(atom_names) == self.n_atoms:
            return True
        return False

    @property
    def improper_torsions(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """A list of improper atom tuples where the first atom is central."""

        improper_torsions = []

        for node in self.topology.nodes:
            near = sorted(list(nx.neighbors(self.topology, node)))
            # if the atom has 3 bonds it could be an improper
            # Check if an sp2 carbon or N
            if len(near) == 3 and (
                self.atoms[node].atomic_symbol == "C"
                or self.atoms[node].atomic_symbol == "N"
            ):
                # Store each combination of the improper torsion
                improper_torsions.append((node, near[0], near[1], near[2]))
        return improper_torsions or None

    @property
    def n_improper_torsions(self) -> int:
        """The number of unique improper torsions."""
        impropers = self.improper_torsions
        if impropers is None:
            return 0
        return len(impropers)

    @property
    def angles(self) -> Optional[List[Tuple[int, int, int]]]:
        """A List of angles from the topology."""

        angles = []

        for node in self.topology.nodes:
            bonded = sorted(list(nx.neighbors(self.topology, node)))

            # Check that the atom has more than one bond
            if len(bonded) < 2:
                continue

            # Find all possible angle combinations from the list
            for i in range(len(bonded)):
                for j in range(i + 1, len(bonded)):
                    atom1, atom3 = bonded[i], bonded[j]
                    angles.append((atom1, node, atom3))
        return angles or None

    @property
    def n_angles(self) -> int:
        """The number of angles in the molecule. """
        angles = self.angles
        if angles is None:
            return 0
        return len(angles)

    def measure_bonds(self, input_type="input") -> Dict[Tuple[int, int], float]:
        """
        Find the length of all bonds in the molecule for the given conformer in  angstroms.

        Returns
        -------
            A dictionary of the bond lengths stored by bond tuple.
        """

        bond_lengths = {}

        molecule = self.coords[input_type]

        for edge in self.topology.edges:
            atom1 = molecule[edge[0]]
            atom2 = molecule[edge[1]]
            bond_lengths[edge] = np.linalg.norm(atom2 - atom1)

        return bond_lengths

    @property
    def bonds(self) -> List[Tuple[int, int]]:
        """A list of bonds from the topology."""
        return list(self.topology.edges)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        bonds = self.bonds
        if bonds is None:
            return 0
        return len(bonds)

    @property
    def dihedrals(
        self,
    ) -> Optional[Dict[Tuple[int, int], List[Tuple[int, int, int, int]]]]:
        """A list of all possible dihedrals that can be found in the topology."""

        dihedrals = {}

        # Work through the network using each edge as a central dihedral bond
        for edge in self.topology.edges:

            for start in list(nx.neighbors(self.topology, edge[0])):

                # Check atom not in main bond
                if start != edge[0] and start != edge[1]:

                    for end in list(nx.neighbors(self.topology, edge[1])):

                        # Check atom not in main bond
                        if end != edge[0] and end != edge[1]:

                            if edge not in dihedrals:
                                # Add the central edge as a key the first time it is used
                                dihedrals[edge] = [(start, edge[0], edge[1], end)]

                            else:
                                # Add the tuple to the correct key.
                                dihedrals[edge].append((start, edge[0], edge[1], end))

        return dihedrals or None

    @property
    def n_dihedrals(self) -> int:
        """The total number of dihedrals in the molecule."""
        dihedrals = self.dihedrals
        if dihedrals is None:
            return 0
        return sum([len(torsions) for torsions in dihedrals.values()])

    @property
    def rotatable_bonds(self) -> Optional[List[Tuple[int, int]]]:
        """
        For each dihedral in the topology graph network and dihedrals dictionary, work out if the torsion is
        rotatable. Returns a list of dihedral dictionary keys representing the rotatable dihedrals.
        Also exclude standard rotations such as amides and methyl groups.
        TODO replace with smarts matching
        """
        dihedrals = self.dihedrals
        if dihedrals is None:
            return None

        rotatable = []

        # For each dihedral key remove the edge from the network
        for key in dihedrals:
            self.topology.remove_edge(*key)

            # Check if there is still a path between the two atoms in the edges.
            if not nx.has_path(self.topology, *key):
                rotatable.append(key)

            # Add edge back to the network and try next key
            self.topology.add_edge(*key)

        remove_list = []
        if rotatable and self.methyl_amine_nitride_cores is not None:
            for key in rotatable:
                if (
                    key[0] in self.methyl_amine_nitride_cores
                    or key[1] in self.methyl_amine_nitride_cores
                ):
                    remove_list.append(key)

            for torsion in remove_list:
                rotatable.remove(torsion)

        return rotatable or None

    @property
    def n_rotatable_bonds(self) -> int:
        """The number of rotatable bonds."""
        rotatable_bonds = self.rotatable_bonds
        if rotatable_bonds is None:
            return 0
        return len(rotatable_bonds)

    def measure_dihedrals(
        self, input_type="input"
    ) -> Optional[Dict[Tuple[int, int, int, int], float]]:
        """
        For the given conformation measure the dihedrals in the topology in degrees.
        """
        dihedrals = self.dihedrals
        if dihedrals is None:
            return None

        dih_phis = {}

        coordinates = self.coords[input_type]

        for val in dihedrals.values():
            for torsion in val:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1, x2, x3, x4 = [coordinates[torsion[i]] for i in range(4)]
                b1, b2, b3 = x2 - x1, x3 - x2, x4 - x3
                t1 = np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3))
                t2 = np.dot(np.cross(b1, b2), np.cross(b2, b3))
                dih_phis[torsion] = np.degrees(np.arctan2(t1, t2))

        return dih_phis

    def measure_angles(
        self, input_type="input"
    ) -> Optional[Dict[Tuple[int, int, int], float]]:
        """
        For the given conformation measure the angles in the topology in degrees.
        """
        angles = self.angles
        if angles is None:
            return None

        angle_values = {}

        coordinates = self.coords[input_type]

        for angle in angles:
            x1 = coordinates[angle[0]]
            x2 = coordinates[angle[1]]
            x3 = coordinates[angle[2]]
            b1, b2 = x1 - x2, x3 - x2
            cosine_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
            angle_values[angle] = np.degrees(np.arccos(cosine_angle))

        return angle_values

    @property
    def n_atoms(self) -> int:
        """
        Calculate the number of atoms.
        """
        return len(self.atoms)

    def write_parameters(
        self, name=None, parameter_tags: Optional[ParameterTags] = None
    ):
        """
        Take the molecule's parameter set and write an xml file for the molecule.
        """

        tree = self.create_forcefield(parameter_tags=parameter_tags).getroot()
        messy = ET.tostring(tree, "utf-8")

        pretty_xml_as_string = parseString(messy).toprettyxml(indent="")

        with open(f"{name if name is not None else self.name}.xml", "w+") as xml_doc:
            xml_doc.write(pretty_xml_as_string)

    def create_forcefield(
        self, parameter_tags: Optional[ParameterTags] = None
    ) -> ET.ElementTree:
        """
        Separates the parameters and builds an xml tree ready to be used.
        """

        # Create XML layout
        root = ET.Element("ForceField")
        AtomTypes = ET.SubElement(root, "AtomTypes")
        Residues = ET.SubElement(root, "Residues")

        Residue = ET.SubElement(
            Residues, "Residue", name=f'{"QUP" if self.is_protein else "UNK"}'
        )

        HarmonicBondForce = ET.SubElement(root, "HarmonicBondForce")
        HarmonicAngleForce = ET.SubElement(root, "HarmonicAngleForce")
        PeriodicTorsionForce = ET.SubElement(root, "PeriodicTorsionForce")

        # Assign the combination rule
        c14 = "0.83333" if self.combination == "amber" else "0.5"
        l14 = "0.5"

        # add the combination rule to the xml for geometric.
        NonbondedForce = ET.SubElement(
            root,
            "NonbondedForce",
            attrib={
                "coulomb14scale": c14,
                "lj14scale": l14,
                "combination": self.combination,
            },
        )

        for key, val in self.AtomTypes.items():
            ET.SubElement(
                AtomTypes,
                "Type",
                attrib={
                    "name": val[1],
                    "class": val[2],
                    "element": self.atoms[key].atomic_symbol,
                    "mass": str(self.atoms[key].atomic_mass),
                },
            )

            ET.SubElement(Residue, "Atom", attrib={"name": val[0], "type": val[1]})

        # Add the bonds / connections
        for key, val in self.HarmonicBondForce.items():
            ET.SubElement(
                Residue, "Bond", attrib={"from": str(key[0]), "to": str(key[1])}
            )
            # build basic data
            bond_data = {
                "class1": self.AtomTypes[key[0]][2],
                "class2": self.AtomTypes[key[1]][2],
                "length": f"{float(val[0]):.6f}",
                "k": f"{float(val[1]):.6f}",
            }
            if parameter_tags is not None:
                if parameter_tags.check_bond_group(bond=key):
                    bond_data.update(parameter_tags.HarmonicBondForce_tags)

            ET.SubElement(
                HarmonicBondForce,
                "Bond",
                attrib=bond_data,
            )

        # Add the angles
        for key, val in self.HarmonicAngleForce.items():
            angle_data = {
                "class1": self.AtomTypes[key[0]][2],
                "class2": self.AtomTypes[key[1]][2],
                "class3": self.AtomTypes[key[2]][2],
                "angle": f"{float(val[0]):.6f}",
                "k": f"{float(val[1]):.6f}",
            }
            if parameter_tags is not None:
                if parameter_tags.check_angle_group(angle=key):
                    angle_data.update(parameter_tags.HarmonicAngleForce_tags)

            ET.SubElement(
                HarmonicAngleForce,
                "Angle",
                attrib=angle_data,
            )

        # add the proper and improper torsion terms
        for key in self.PeriodicTorsionForce:
            if self.PeriodicTorsionForce[key][-1] == "Improper":
                tor_type = "Improper"
            else:
                tor_type = "Proper"

            torsion_data = {
                "class1": self.AtomTypes[key[0]][2],
                "class2": self.AtomTypes[key[1]][2],
                "class3": self.AtomTypes[key[2]][2],
                "class4": self.AtomTypes[key[3]][2],
                "k1": str(self.PeriodicTorsionForce[key][0][1]),
                "k2": str(self.PeriodicTorsionForce[key][1][1]),
                "k3": str(self.PeriodicTorsionForce[key][2][1]),
                "k4": str(self.PeriodicTorsionForce[key][3][1]),
                "periodicity1": "1",
                "periodicity2": "2",
                "periodicity3": "3",
                "periodicity4": "4",
                "phase1": str(self.PeriodicTorsionForce[key][0][2]),
                "phase2": str(self.PeriodicTorsionForce[key][1][2]),
                "phase3": str(self.PeriodicTorsionForce[key][2][2]),
                "phase4": str(self.PeriodicTorsionForce[key][3][2]),
            }
            if parameter_tags is not None:
                if parameter_tags.check_torsion_group(torsion=key):
                    torsion_data.update(parameter_tags.PeriodicTorsionForce_tags)
                    # we should also remove and 0 k values for forcebalance
                    for i in range(1, 5):
                        k_val = f"k{i}"
                        if torsion_data[k_val] == "0":
                            torsion_data[k_val] = "1e-6"

            ET.SubElement(
                PeriodicTorsionForce,
                tor_type,
                attrib=torsion_data,
            )

        # add the non-bonded parameters
        for key in self.NonbondedForce:
            ET.SubElement(
                NonbondedForce,
                "Atom",
                attrib={
                    "type": self.AtomTypes[key][1],
                    "charge": f"{self.NonbondedForce[key][0]:.6f}",
                    "sigma": f"{self.NonbondedForce[key][1]:.6f}",
                    "epsilon": f"{self.NonbondedForce[key][2]:.6f}",
                },
            )

        # Add all of the virtual site info if present
        if self.extra_sites is not None:
            # Add the atom type to the top
            for key, site in self.extra_sites.items():
                ET.SubElement(
                    AtomTypes,
                    "Type",
                    attrib={
                        "name": f"v-site{key + 1}",
                        "class": f"X{key + 1}",
                        "mass": "0",
                    },
                )

                # Add the atom info
                ET.SubElement(
                    Residue,
                    "Atom",
                    attrib={"name": f"X{key + 1}", "type": f"v-site{key + 1}"},
                )

                # Add the local coords site info
                attrib = {
                    "type": "localCoords",
                    "index": str(key + len(self.atoms)),
                    "atom1": str(site.parent_index),
                    "atom2": str(site.closest_a_index),
                    "atom3": str(site.closest_b_index),
                    "wo1": str(site.o_weights[0]),
                    "wo2": str(site.o_weights[1]),
                    "wo3": str(site.o_weights[2]),
                    "wx1": str(site.x_weights[0]),
                    "wx2": str(site.x_weights[1]),
                    "wx3": str(site.x_weights[2]),
                    "wy1": str(site.y_weights[0]),
                    "wy2": str(site.y_weights[1]),
                    "wy3": str(site.y_weights[2]),
                    "p1": str(site.p1),
                    "p2": str(site.p2),
                    "p3": str(site.p3),
                }

                # For the Nitrogen case
                if len(site.o_weights) == 4:
                    attrib["wo4"] = str(site.o_weights[3])
                    attrib["wx4"] = str(site.x_weights[3])
                    attrib["wy4"] = str(site.y_weights[3])
                    attrib["atom4"] = str(site.closest_c_index)

                ET.SubElement(Residue, "VirtualSite", attrib=attrib)

                # Add the nonbonded info
                ET.SubElement(
                    NonbondedForce,
                    "Atom",
                    attrib={
                        "type": f"v-site{key + 1}",
                        "charge": str(site.charge),
                        "sigma": "1.000000",
                        "epsilon": "0.000000",
                    },
                )

        # Store the tree back into the molecule
        return ET.ElementTree(root)

    def write_xyz(self, input_type="input", name=None):
        """
        Write a general xyz file of the molecule if there are multiple geometries in the molecule write a traj
        :param input_type: Where the molecule coordinates are taken from
        :param name: The name of the xyz file to be produced; otherwise self.name is used.
        """

        with open(f"{name if name is not None else self.name}.xyz", "w+") as xyz_file:

            if len(self.coords[input_type]) == len(self.atoms):
                message = "xyz file generated with QUBEKit"
                end = ""
                trajectory = [self.coords[input_type]]

            else:
                message = "QUBEKit xyz trajectory FRAME "
                end = 1
                trajectory = self.coords[input_type]

            # Write out each frame
            for frame in trajectory:

                xyz_file.write(f"{len(self.atoms)}\n")
                xyz_file.write(f"{message}{end}\n")

                for i, atom in enumerate(frame):
                    xyz_file.write(
                        f"{self.atoms[i].atomic_symbol}       {atom[0]: .10f}   {atom[1]: .10f}   {atom[2]: .10f}\n"
                    )

                try:
                    end += 1
                except TypeError:
                    # This is the result of only printing one frame so catch the error and ignore
                    pass

    def pickle(self, state=None):
        """
        Pickles the Molecule object in its current state to the (hidden) pickle file.
        If other pickle objects already exist for the particular object:
            the latest object is put to the top.
        """

        mols = OrderedDict()
        # First check if the pickle file exists
        try:
            # Try to load a hidden pickle file; make sure to get all objects
            with open(".QUBEKit_states", "rb") as pickle_jar:
                while True:
                    try:
                        mol = pickle.load(pickle_jar)
                        mols[mol.state] = mol
                    except EOFError:
                        break
        except FileNotFoundError:
            pass

        # Now we can save the items; first assign the location
        self.state = state
        mols[self.state] = self

        # Open the pickle jar which will always be the ligand object's name
        with open(".QUBEKit_states", "wb") as pickle_jar:

            # If there were other molecules of the same state in the jar: overwrite them
            for val in mols.values():
                pickle.dump(val, pickle_jar)

    @property
    def bond_types(self) -> Dict[str, Tuple[int, int]]:
        """
        Using the symmetry dict, give each bond a code. If any codes match, the bonds can be symmetrised.
        e.g. bond_symmetry_classes = {(0, 3): '2-0', (0, 4): '2-0', (0, 5): '2-0' ...}
        all of the above bonds (tuples) are of the same type (methyl H-C bonds in same region)
        This dict is then used to produce bond_types.
        bond_types is just a dict where the keys are the string code from above and the values are all
        of the bonds with that particular type.
        """
        atom_types = self.atom_types
        bond_symmetry_classes = {}
        for bond in self.bonds:
            bond_symmetry_classes[bond] = (
                f"{atom_types[bond[0]]}-" f"{atom_types[bond[1]]}"
            )

        bond_types = {}
        for key, val in bond_symmetry_classes.items():
            bond_types.setdefault(val, []).append(key)

        bond_types = self._cluster_types(bond_types)
        return bond_types

    @property
    def angle_types(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Using the symmetry dict, give each angle a code. If any codes match, the angles can be symmetrised.
        e.g. angle_symmetry_classes = {(1, 0, 3): '3-2-0', (1, 0, 4): '3-2-0', (1, 0, 5): '3-2-0' ...}
        all of the above angles (tuples) are of the same type (methyl H-C-H angles in same region)
        angle_types is just a dict where the keys are the string code from the above and the values are all
        of the angles with that particular type.
        """
        atom_types = self.atom_types
        angle_symmetry_classes = {}
        for angle in self.angles:
            angle_symmetry_classes[angle] = (
                f"{atom_types[angle[0]]}-"
                f"{atom_types[angle[1]]}-"
                f"{atom_types[angle[2]]}"
            )

        angle_types = {}
        for key, val in angle_symmetry_classes.items():
            angle_types.setdefault(val, []).append(key)

        angle_types = self._cluster_types(angle_types)
        return angle_types

    @property
    def dihedral_types(self) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Using the symmetry dict, give each dihedral a code. If any codes match, the dihedrals can be clustered and their
        parameters should be the same, this is to be used in dihedral fitting so all symmetry equivalent dihedrals are
        optimised at the same time. dihedral_equiv_classes = {(0, 1, 2 ,3): '1-1-2-1'...} all of the tuples are the
        dihedrals index by topology and the strings are the symmetry equivalent atom combinations.
        """
        atom_types = self.atom_types
        dihedral_symmetry_classes = {}
        for dihedral_set in self.dihedrals.values():
            for dihedral in dihedral_set:
                dihedral_symmetry_classes[tuple(dihedral)] = (
                    f"{atom_types[dihedral[0]]}-"
                    f"{atom_types[dihedral[1]]}-"
                    f"{atom_types[dihedral[2]]}-"
                    f"{atom_types[dihedral[3]]}"
                )

        dihedral_types = {}
        for key, val in dihedral_symmetry_classes.items():
            dihedral_types.setdefault(val, []).append(key)

        dihedral_types = self._cluster_types(dihedral_types)
        return dihedral_types

    @property
    def improper_types(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Using the atom symmetry types work out the improper types."""

        atom_types = self.atom_types
        improper_symmetry_classes = {}
        for dihedral in self.improper_torsions:
            improper_symmetry_classes[tuple(dihedral)] = (
                f"{atom_types[dihedral[0]]}-"
                f"{atom_types[dihedral[1]]}-"
                f"{atom_types[dihedral[2]]}-"
                f"{atom_types[dihedral[3]]}"
            )

        improper_types = {}
        for key, val in improper_symmetry_classes.items():
            improper_types.setdefault(val, []).append(key)

        improper_types = self._cluster_types(improper_types)
        return improper_types

    @staticmethod
    def _cluster_types(equiv_classes):
        """
        Function that helps the bond angle and dihedral class finders in clustering the types based on the forward and
        backward type strings.
        :return: clustered equiv class
        """

        new_classes = {}
        for key, item in equiv_classes.items():
            try:
                new_classes[key].extend(item)
            except KeyError:
                try:
                    new_classes[key[::-1]].extend(item)
                except KeyError:
                    new_classes[key] = item

        return new_classes

    @property
    def atom_types(self) -> Optional[Dict[int, str]]:
        """Returns a dictionary of atom indices mapped to their class or None if there is no rdkit molecule.
        #TODO we need a to_rdkit method as this should always work for well defined inputs.
        """
        if self.rdkit_mol is not None:
            return RDKit.find_symmetry_classes(self.rdkit_mol)
        return None

    def symmetrise_from_topology(self) -> None:
        """
        First, if rdkit_mol has been generated, get the bond and angle symmetry dicts.
        These will be used by L-J and the Harmonic Bond/Angle params

        Then, based on the molecule topology, symmetrise the methyl / amine hydrogens.
        If there's a carbon, does it have 3/2 hydrogens? -> symmetrise
        If there's a nitrogen, does it have 2 hydrogens? -> symmetrise
        Also keep a list of the methyl carbons and amine / nitrile nitrogens
        then exclude these bonds from the rotatable torsions list.

        TODO This needs to be more applicable to proteins (e.g. if no rdkit_mol is created).
        """

        methyl_hs, amine_hs, other_hs = [], [], []
        methyl_amine_nitride_cores = []

        for atom in self.atoms:
            if atom.atomic_symbol == "C" or atom.atomic_symbol == "N":

                hs = []
                for bonded in self.topology.neighbors(atom.atom_index):
                    if len(list(self.topology.neighbors(bonded))) == 1:
                        # now make sure it is a hydrogen (as halogens could be caught here)
                        if self.atoms[bonded].atomic_symbol == "H":
                            hs.append(bonded)

                if (
                    atom.atomic_symbol == "C" and len(hs) == 2
                ):  # This is part of a carbon hydrogen chain
                    other_hs.append(hs)
                elif atom.atomic_symbol == "C" and len(hs) == 3:
                    methyl_hs.append(hs)
                    methyl_amine_nitride_cores.append(atom.atom_index)
                elif atom.atomic_symbol == "N" and len(hs) == 2:
                    amine_hs.append(hs)
                    methyl_amine_nitride_cores.append(atom.atom_index)

        self.symm_hs = {"methyl": methyl_hs, "amine": amine_hs, "other": other_hs}
        self.methyl_amine_nitride_cores = methyl_amine_nitride_cores

    def openmm_coordinates(self, input_type="input") -> List[np.array]:
        """
        Take a set of coordinates from the molecule and convert them to OpenMM format
        :param input_type: The set of coordinates that should be used
        :return: A list of tuples of the coords
        TODO Move elsewhere; currently breaks
        """

        coordinates = self.coords[input_type]

        # Multiple frames in this case
        if input_type == "traj" and len(coordinates) != self.n_atoms:
            return [frame / 10 for frame in coordinates]
        return [coordinates / 10]

    def read_tdrive(self, bond_scan):
        """
        Read a tdrive qdata file and get the coordinates and scan energies and store in the molecule.
        :type bond_scan: the tuple of the scanned central bond
        :return: None, store the coords in the traj holder and the energies in the qm scan holder
        TODO Move elsewhere
        """

        scan_coords = []
        energy = []
        qm_scans = {}
        with open("qdata.txt", "r") as data:
            for line in data.readlines():
                if "COORDS" in line:
                    coords = [float(x) for x in line.split()[1:]]
                    coords = np.array(coords).reshape((len(self.atoms), 3))
                    scan_coords.append(coords)
                elif "ENERGY" in line:
                    energy.append(float(line.split()[1]))

        qm_scans[bond_scan] = [np.array(energy), scan_coords]
        if self.qm_scans is not None:
            self.qm_scans = {**self.qm_scans, **qm_scans}
        else:
            self.qm_scans = qm_scans or None

    def read_scan_order(self, file):
        """
        Read a QUBEKit or tdrive dihedrals file and store the scan order into the ligand class
        :param file: The dihedrals input file.
        :return: The molecule with the scan_order saved
        TODO Move elsewhere
        """

        # If we have a QUBE.dihedrals file get the scan order from there
        scan_order = []
        torsions = open(file).readlines()
        for line in torsions:
            if "#" not in line:
                torsion = line.split()
                if len(torsion) == 6:
                    print("Torsion and dihedral range found, updating scan range:")
                    # TODO Why are these class attributes?
                    self.dih_start = int(torsion[-2])
                    self.dih_end = int(torsion[-1])
                    print(
                        f"Dihedral will be scanned in the range: {self.dih_start},  {self.dih_end}"
                    )
                core = (int(torsion[1]), int(torsion[2]))
                if core in self.dihedrals.keys():
                    scan_order.append(core)
                elif reversed(tuple(core)) in self.dihedrals.keys():
                    scan_order.append(reversed(tuple(core)))
                else:
                    # This might be an improper scan so check
                    improper = (
                        int(torsion[0]),
                        int(torsion[1]),
                        int(torsion[2]),
                        int(torsion[3]),
                    )
                    if improper in self.improper_torsions:
                        print("Improper torsion found.")
                        scan_order.append(improper)
        self.scan_order = scan_order


class Ligand(DefaultsMixin, Molecule):
    def __init__(self, mol_input, name=None):
        """
        parameter_engine        A string keeping track of the parameter engine used to assign the initial parameters
        hessian                 2d numpy array; matrix of size 3N x 3N where N is number of atoms in the molecule
        modes                   A list of the qm predicted frequency modes
        home

        constraints_file        Either an empty string (does nothing in geometric run command); or
                                the abspath of the constraint.txt file (constrains the execution of geometric)
        """

        super().__init__(mol_input, name)

        self.is_protein = False

        self.parameter_engine = "openmm"
        self.hessian = None
        self.modes = None
        self.home = None

        # Charge and LJ data from Chargemol / ONETEP
        self.ddec_data = None
        self.dipole_moment_data = None
        self.quadrupole_moment_data = None
        self.cloud_pen_data = None

        self.constraints_file = None

        # if this is not the readinput data we assume its a file only
        if not isinstance(mol_input, ReadInput):
            self._check_file_name(file_name=mol_input)
            input_data = ReadInput.from_file(file_name=self.mol_input)
        else:
            input_data = mol_input
        self._save_to_ligand(mol_input=input_data)

        # Make sure we have the topology before we calculate the properties
        if self.topology.edges:
            self.symmetrise_from_topology()

        # make sure we have unique atom names
        self._validate_atom_names()

    @classmethod
    def from_rdkit(cls, rdkit_mol, name: Optional[str] = None) -> "Ligand":
        """
        Build an instance of a qubekit ligand directly from an rdkit molecule.
        """
        input_data = ReadInput.from_rdkit(rdkit_mol=rdkit_mol)
        ligand = cls(mol_input=input_data, name=name)
        return ligand

    @staticmethod
    def _check_file_name(file_name):
        """
        Make sure that if an unsupported file type is passed we can not make a molecule from it.
        """
        if ".xyz" in file_name:
            raise FileTypeError(
                "XYZ files can not be used to build ligands due to ambiguous bonding, please use pdb, mol, mol2 or smiles as input."
            )

    @classmethod
    def from_file(cls, file_name: str) -> "Ligand":
        """
        Build a ligand from an input file.
        """
        cls._check_file_name(file_name=file_name)
        input_data = ReadInput.from_file(file_name=file_name)
        ligand = cls(mol_input=input_data)
        return ligand

    @classmethod
    def from_smiles(cls, smiles_string: str, name: str):
        """
        Build the ligand molecule directly from the smiles string.
        """
        input_data = ReadInput.from_smiles(smiles=smiles_string, name=name)
        ligand = cls(mol_input=input_data)
        return ligand

    def generate_atom_names(self) -> None:
        """
        Generate a unique set of atom names for the molecule.
        """
        atom_names = {}
        for atom in self.atoms:
            symbol = atom.atomic_symbol
            if symbol not in atom_names:
                atom_names[symbol] = 1
            else:
                atom_names[symbol] += 1

            atom.atom_name = f"{symbol}{atom_names[symbol]}"

    def _validate_atom_names(self) -> None:
        """
        Check that the ligand has unique atom names if not generate a new set.
        """
        if not self.has_unique_atom_names:
            self.generate_atom_names()

    def add_conformers(self, file_name: str, input_type="input") -> None:
        """
        Read the given input file extract  the conformers and save them to the ligand.
        #TODO do we want to check that the conectivity is the same?
        """
        input_data = ReadInput.from_file(file_name=file_name)
        self.coords[input_type] = input_data.coords

    def _save_to_ligand(self, mol_input: ReadInput, input_type="input") -> None:
        """
        Public access to private file_handlers.py file.
        Users shouldn't ever need to interface with file_handlers.py directly.
        All parameters will be set from a file (or other input) via this public method.

        Don't bother updating name, topology or atoms if they are already stored.
        Do bother updating coords and rdkit_mol

        :param mol_input:
        :param name:
        :param input_type: "input", "mm", "qm", "traq", or "temp"
        """

        if mol_input.name is not None:
            self.name = mol_input.name
        if mol_input.topology is not None:
            self.topology = mol_input.topology
        if mol_input.atoms is not None:
            self.atoms = mol_input.atoms
        if mol_input.coords is not None:
            self.coords[input_type] = mol_input.coords
        if mol_input.rdkit_mol is not None:
            self.rdkit_mol = mol_input.rdkit_mol

    def write_pdb(self, input_type="input", name=None):
        """
        Take the current molecule and topology and write a pdb file for the molecule.
        Only for small molecules, not standard residues. No size limit.
        """

        molecule = self.coords[input_type]

        with open(f"{name if name is not None else self.name}.pdb", "w+") as pdb_file:

            # Write out the atomic xyz coordinates
            pdb_file.write(f"REMARK   1 CREATED WITH QUBEKit {datetime.now()}\n")
            pdb_file.write(f"COMPND    {self.name:<20}\n")
            for i, atom in enumerate(molecule):
                pdb_file.write(
                    f"HETATM {i+1:>4}{self.atoms[i].atom_name:>4}  UNL     1{atom[0]:12.3f}{atom[1]:8.3f}{atom[2]:8.3f}"
                    f"  1.00  0.00         {self.atoms[i].atomic_symbol.title():>3}\n"
                )

            # Now add the connection terms
            for node in self.topology.nodes:
                bonded = sorted(list(nx.neighbors(self.topology, node)))
                if len(bonded) > 1:
                    pdb_file.write(
                        f'CONECT{node + 1:5}{"".join(f"{x + 1:5}" for x in bonded)}\n'
                    )

            pdb_file.write("END\n")


class Protein(DefaultsMixin, Molecule):
    """
    This class handles the protein input to make the QUBEKit xml files and rewrite the pdb so we can use it.
    """

    def __init__(self, mol_input, name=None):
        """
        is_protein      Bool; True for Protein class
        home            Current working directory (location for QUBEKit execution).
        residues        List of all residues in the molecule in order e.g. ['ARG', 'HIS', ... ]
        Residues        List of residue names for each atom e.g. ['ARG', 'ARG', 'ARG', ... 'HIS', 'HIS', ... ]
        pdb_names       List
        """

        super().__init__(mol_input, name)

        self.is_protein = True
        self.home = os.getcwd()
        self.residues = None
        self.Residues = None
        self.pdb_names = None

        self.combination = "opls"

        if not isinstance(mol_input, ReadInputProtein):
            self._check_file_type(file_name=mol_input)
            input_data = ReadInputProtein.from_pdb(file_name=mol_input)
        else:
            input_data = mol_input
        self._save_to_protein(input_data, input_type="input")

    @classmethod
    def from_file(cls, file_name: str, name: Optional[str] = None) -> "Protein":
        """
        Instance the protein class from a pdb file.
        """
        cls._check_file_type(file_name=file_name)
        input_data = ReadInputProtein.from_pdb(file_name=file_name, name=name)
        return cls(input_data)

    @staticmethod
    def _check_file_type(file_name: str) -> None:
        """
        Make sure the protien is being read from a pdb file.
        """
        if ".pdb" not in file_name:
            raise FileTypeError("Proteins can only be read from pdb.")

    def _save_to_protein(self, mol_input: ReadInputProtein, input_type="input"):
        """
        Public access to private file_handlers.py file.
        Users shouldn't ever need to interface with file_handlers.py directly.
        All parameters will be set from a file (or other input) via this public method.
            * Don't bother updating name, topology or atoms if they are already stored.
            * Do bother updating coords, rdkit_mol, residues, Residues, pdb_names
        """

        if mol_input.name is not None:
            self.name = mol_input.name
        if mol_input.topology is not None:
            self.topology = mol_input.topology
        if mol_input.atoms is not None:
            self.atoms = mol_input.atoms
        if mol_input.coords is not None:
            self.coords[input_type] = mol_input.coords
        if mol_input.residues is not None:
            self.residues = mol_input.residues
        if mol_input.pdb_names is not None:
            self.pdb_names = mol_input.pdb_names

        if not self.topology.edges:
            print(
                "No connections found in pdb file; topology will be inferred by OpenMM."
            )
            return

        # self.find_angles()
        # self.find_dihedrals()
        # self.find_rotatable_dihedrals()
        # self.find_impropers()
        # self.measure_dihedrals(input_type)
        # self.bond_lengths(input_type)
        # self.measure_angles(input_type)
        # This creates the dictionary of terms that should be symmetrised.
        self.symmetrise_from_topology()

    def write_pdb(self, name=None):
        """
        This method replaces the ligand method as all of the atom names and residue names have to be replaced.
        """

        with open(f"{name if name is not None else self.name}.pdb", "w+") as pdb_file:

            pdb_file.write(f"REMARK   1 CREATED WITH QUBEKit {datetime.now()}\n")
            # Write out the atomic xyz coordinates
            for i, (coord, atom) in enumerate(zip(self.coords["input"], self.atoms)):
                x, y, z = coord
                # May cause issues if protein contains more than 10,000 atoms.
                pdb_file.write(
                    f"HETATM {i+1:>4}{atom.atom_name:>5} QUP     1{x:12.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00         {atom.atomic_symbol.upper():>3}\n"
                )

            # Add the connection terms based on the molecule topology.
            for node in self.topology.nodes:
                bonded = sorted(list(nx.neighbors(self.topology, node)))
                if len(bonded) >= 1:
                    pdb_file.write(
                        f'CONECT{node + 1:5}{"".join(f"{x + 1:5}" for x in bonded)}\n'
                    )

            pdb_file.write("END\n")

    def update(self, input_type="input"):
        """
        After the protein has been passed to the parametrisation class we get back the bond info
        use this to update all missing terms.
        """

        # using the new harmonic bond force dict we can add the bond edges to the topology graph
        for bond in self.HarmonicBondForce:
            self.topology.add_edge(*bond)

        # self.find_angles()
        # self.find_dihedrals()
        # self.find_rotatable_dihedrals()
        # self.find_impropers()
        # self.measure_dihedrals(input_type)
        # self.bond_lengths(input_type)
        # self.measure_angles(input_type)
        # This creates the dictionary of terms that should be symmetrised.
        self.symmetrise_from_topology()
