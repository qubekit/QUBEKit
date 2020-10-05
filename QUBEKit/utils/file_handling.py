#!/usr/bin/env python3

"""
Purpose of this file is to read various inputs and produce the info required for
    Ligand() or Protein()
Should be very little calculation here, simply file reading and some small validations / checks

TODO
    Need to re-add topology checking and name checking (Do this in ligand.py?)
    Descriptors should be accessed separately if needed (need to re-add)
"""

from QUBEKit.engines import RDKit
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import Atom, CustomNamespace, Element
from QUBEKit.utils.exceptions import FileTypeError

from itertools import groupby
import os
from pathlib import Path
import re

import networkx as nx
import numpy as np


class ReadInput:
    """
    Called inside Ligand or Protein; used to handle reading any kind of input valid in QUBEKit
        QC JSON object
        SMILES string
        PDB, MOL2, XYZ file
    :param mol_input: One of the accepted input types:
        QC JSON object
        SMILES string
        PDB, MOL2, XYZ file
    :param name: The name of the molecule. Only necessary for smiles strings but can be
        provided regardless of input type.
    """

    def __init__(self, mol_input, name=None, is_protein=False):

        if Path(mol_input).exists():
            self.mol_input = Path(mol_input)
            if name is None:
                self.name = self.mol_input.stem
            else:
                self.name = name
        else:
            self.mol_input = mol_input
            self.name = name

        self.topology = None
        self.atoms = None
        self.coords = None

        self.rdkit_mol = None

        if is_protein:
            self._read_pdb_protein()
        else:
            self._read_input()

    def _read_input(self):
        """
        Figure out what the input is (file, smiles, json) and call the relevant method.
        """

        if self.mol_input.__class__.__name__ == 'Molecule':
            # QCArchive object
            self._read_qc_json()

        elif hasattr(self.mol_input, 'stem'):
            # File (pdb, xyz, etc)
            try:
                # Try parse with rdkit:
                self.rdkit_mol = RDKit.mol_input_to_rdkit_mol(self.mol_input)
                self._mol_from_rdkit()
            except AttributeError:
                # Cannot be parsed by rdkit:
                self._read_file()

        elif isinstance(self.mol_input, str):
            # Smiles string input
            self.rdkit_mol = RDKit.smiles_to_rdkit_mol(self.mol_input, self.name)
            self._mol_from_rdkit()

        else:
            raise RuntimeError('Cannot read input. mol_input must be a smiles string, path of a file, or qc json.')

    def _read_file(self):
        """
        Called when rdkit cannot parse the file.
        Calls the necessary internal file readers instead.
        """

        if self.mol_input.suffix == '.pdb':
            self._read_pdb()
        elif self.mol_input.suffix == '.mol2':
            self._read_mol2()
        elif self.mol_input.suffix == '.xyz':
            self._read_xyz()
        else:
            raise FileTypeError(f'Could not read file {self.mol_input}. File type must be pdb, mol2 or xyz.')

    def _mol_from_rdkit(self):
        """
        Using an RDKit Molecule object, extract the name, topology, coordinates and atoms
        """
        if self.name is None:
            self.name = self.rdkit_mol.GetProp('_Name')

        atoms = []
        self.topology = nx.Graph()
        # Collect the atom names and bonds
        for atom in self.rdkit_mol.GetAtoms():
            # Collect info about each atom
            atomic_number = atom.GetAtomicNum()
            index = atom.GetIdx()
            try:
                # PDB file extraction
                atom_name = atom.GetMonomerInfo().GetName().strip()
            except AttributeError:
                try:
                    # Mol2 file extraction
                    atom_name = atom.GetProp('_TriposAtomName')
                except KeyError:
                    # smiles and mol files have no atom names so generate them here if they are not declared
                    atom_name = f'{atom.GetSymbol()}{index}'

            qube_atom = Atom(atomic_number, index, atom_name, formal_charge=atom.GetFormalCharge())

            # Instance the basic qube_atom
            qube_atom.atom_type = atom.GetSmarts()

            # Add the atoms as nodes
            self.topology.add_node(atom.GetIdx())

            # Add the bonds
            for bonded in atom.GetNeighbors():
                self.topology.add_edge(atom.GetIdx(), bonded.GetIdx())
                qube_atom.add_bond(bonded.GetIdx())

            # Now add the atom to the molecule
            atoms.append(qube_atom)

        self.coords = self.rdkit_mol.GetConformer().GetPositions()
        self.atoms = atoms or None

    def _read_qc_json(self):
        """
        Given a QC JSON object, extracts the topology, atoms and coords of the molecule.
        """

        self.topology = nx.Graph()
        atoms = []

        for i, atom in enumerate(self.mol_input.symbols):
            atoms.append(Atom(atomic_number=Element().number(atom), atom_index=i, atom_name=f'{atom}{i}'))
            self.topology.add_node(i)

        for bond in self.mol_input.connectivity:
            self.topology.add_edge(*bond[:2])

        self.coords = np.array(self.mol_input.geometry).reshape((len(atoms), 3)) * constants.BOHR_TO_ANGS
        self.atoms = atoms or None

    def _read_pdb(self):
        """
        Internal pdb reader. Only called when RDKit failed to read the pdb.
        Extracts the topology, atoms and coords of the molecule.
        """

        coords = []
        self.topology = nx.Graph()
        atoms = []

        atom_count = 0

        with open(self.mol_input) as pdb:

            for line in pdb:
                if 'ATOM' in line or 'HETATM' in line:
                    # start collecting the atom class info
                    atomic_symbol = str(line[76:78])
                    atomic_symbol = re.sub('[0-9]+', '', atomic_symbol)
                    atomic_symbol = atomic_symbol.strip()
                    atom_name = str(line.split()[2])

                    # If the element column is missing from the pdb, extract the atomic_symbol from the atom name.
                    if not atomic_symbol:
                        atomic_symbol = str(line.split()[2])[:-1]
                        atomic_symbol = re.sub('[0-9]+', '', atomic_symbol)

                    atomic_number = Element().number(atomic_symbol)
                    # Now instance the qube atom
                    qube_atom = Atom(atomic_number, atom_count, atom_name)
                    atoms.append(qube_atom)

                    # Also add the atom number as the node in the graph
                    self.topology.add_node(atom_count)
                    atom_count += 1
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

                if 'CONECT' in line:
                    atom_index = int(line.split()[1]) - 1
                    # Search the connectivity section and add all edges to the graph corresponding to the bonds.
                    for i in range(2, len(line.split())):
                        if int(line.split()[i]) != 0:
                            bonded_index = int(line.split()[i]) - 1
                            self.topology.add_edge(atom_index, bonded_index)
                            atoms[atom_index].add_bond(bonded_index)
                            atoms[bonded_index].add_bond(atom_index)

        # put the object back into the correct place
        self.coords = np.array(coords)
        self.atoms = atoms or None

    def _read_mol2(self):
        """
        Internal mol2 reader. Only called when RDKit failed to read the mol2.
        Extracts the topology, atoms and coords of the molecule.
        """

        coords = []
        self.topology = nx.Graph()
        atoms = []

        atom_count = 0

        with open(self.mol_input, 'r') as mol2:

            atom_flag = False
            bond_flag = False

            for line in mol2:
                if '@<TRIPOS>ATOM' in line:
                    atom_flag = True
                    continue
                elif '@<TRIPOS>BOND' in line:
                    atom_flag = False
                    bond_flag = True
                    continue
                elif '@<TRIPOS>SUBSTRUCTURE' in line:
                    bond_flag = False
                    continue

                if atom_flag:
                    # Add the molecule information
                    atomic_symbol = line.split()[1][:2]
                    atomic_symbol = re.sub('[0-9]+', '', atomic_symbol)
                    atomic_symbol = atomic_symbol.strip().title()

                    atomic_number = Element().number(atomic_symbol)

                    coords.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])

                    # Collect the atom names
                    atom_name = str(line.split()[1])

                    # Add the nodes to the topology object
                    self.topology.add_node(atom_count)
                    atom_count += 1

                    # Get the atom types
                    atom_type = line.split()[5]
                    atom_type = atom_type.replace(".", "")

                    # Make the qube_atom
                    qube_atom = Atom(atomic_number, atom_count, atom_name)
                    qube_atom.atom_type = atom_type

                    atoms.append(qube_atom)

                if bond_flag:
                    # Add edges to the topology network
                    atom_index, bonded_index = int(line.split()[1]) - 1, int(line.split()[2]) - 1
                    self.topology.add_edge(atom_index, bonded_index)
                    atoms[atom_index].add_bond(bonded_index)
                    atoms[bonded_index].add_bond(atom_index)

        # put the object back into the correct place
        self.coords = np.array(coords)
        self.atoms = atoms or None

    def _read_xyz(self):
        """
        Internal xyz reader.
        Extracts the coords of the molecule.
        """

        traj_molecules = []
        coords = []

        with open(self.mol_input) as xyz_file:
            lines = xyz_file.readlines()

            n_atoms = float(lines[0])

            for line in lines:
                line = line.split()
                # skip frame heading lines
                if len(line) <= 1 or 'Iteration' in line:
                    continue

                coords.append([float(line[1]), float(line[2]), float(line[3])])

                if len(coords) == n_atoms:
                    # we have collected the molecule now store the frame
                    traj_molecules.append(np.array(coords))
                    coords = []

        self.coords = traj_molecules[0] if len(traj_molecules) == 1 else traj_molecules

    def _read_pdb_protein(self):
        """

        :return:
        """
        with open(self.mol_input, 'r') as pdb:
            lines = pdb.readlines()

        coords = []
        atoms = []
        self.topology = nx.Graph()
        self.Residues = []
        self.pdb_names = []

        # atom counter used for graph node generation
        atom_count = 0
        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                atomic_symbol = str(line[76:78])
                atomic_symbol = re.sub('[0-9]+', '', atomic_symbol).strip()

                # If the element column is missing from the pdb, extract the atomic_symbol from the atom name.
                if not atomic_symbol:
                    atomic_symbol = str(line.split()[2])
                    atomic_symbol = re.sub('[0-9]+', '', atomic_symbol)

                # now make sure we have a valid element
                if atomic_symbol.lower() != 'cl' and atomic_symbol.lower() != 'br':
                    atomic_symbol = atomic_symbol[0]

                atom_name = f'{atomic_symbol}{atom_count}'
                qube_atom = Atom(Element().number(atomic_symbol), atom_count, atom_name)

                atoms.append(qube_atom)

                self.pdb_names.append(str(line.split()[2]))

                # also get the residue order from the pdb file so we can rewrite the file
                self.Residues.append(str(line.split()[3]))

                # Also add the atom number as the node in the graph
                self.topology.add_node(atom_count)
                atom_count += 1
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

            elif 'CONECT' in line:
                conect_terms = line.split()
                for atom in conect_terms[2:]:
                    if int(atom):
                        self.topology.add_edge(int(conect_terms[1]) - 1, int(atom) - 1)

        self.atoms = atoms
        self.coords = np.array(coords)
        self.residues = [res for res, group in groupby(self.Residues)]


class ExtractChargeData:
    """
    Choose between extracting (the more extensive) data from Chargemol, or from ONETEP.
    Store all info back into the molecule object.
    """
    def __init__(self, molecule):
        self.molecule = molecule

    def extract_charge_data(self):
        if self.molecule.charges_engine.casefold() == 'chargemol':
            ddec_data, dipole_moment_data, quadrupole_moment_data = self._extract_charge_data_chargemol()
        else:
            ddec_data = self._extract_charge_data_onetep()
            dipole_moment_data = quadrupole_moment_data = None

        self.molecule.ddec_data = ddec_data
        self.molecule.dipole_moment_data = dipole_moment_data
        self.molecule.quadrupole_moment_data = quadrupole_moment_data

        # Ensure the partial charges in the atom container are also changed.
        for molecule_atom, ddec_atom in zip(self.molecule.atoms, self.molecule.ddec_data):
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
            net_charge_file_name = 'DDEC6_even_tempered_net_atomic_charges.xyz'

        elif self.molecule.ddec_version == 3:
            net_charge_file_name = 'DDEC3_net_atomic_charges.xyz'

        else:
            raise ValueError('Unsupported DDEC version; please use version 3 or 6.')

        if not os.path.exists(net_charge_file_name):
            raise FileNotFoundError(
                'Cannot find the DDEC output file.\nThis could be indicative of several issues.\n'
                'Please check Chargemol is installed in the correct location and that the configs'
                ' point to that location.'
            )

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

        ddec_data = {}
        dipole_moment_data = {}
        quadrupole_moment_data = {}

        for line in lines[start_pos: start_pos + atom_total]:
            # _'s are the xyz coords, then the quadrupole moment tensor eigenvalues
            atom_count, atomic_symbol, _, _, _, charge, x_dipole, y_dipole, z_dipole, _, q_xy, q_xz, q_yz, q_x2_y2, q_3z2_r2, *_ = line.split()
            # File counts from 1 not 0; thereby requiring -1 to get the index
            atom_index = int(atom_count) - 1
            ddec_data[atom_index] = CustomNamespace(
                atomic_symbol=atomic_symbol, charge=float(charge), volume=None, r_aim=None, b_i=None, a_i=None
            )

            dipole_moment_data[atom_index] = CustomNamespace(
                x_dipole=float(x_dipole), y_dipole=float(y_dipole), z_dipole=float(z_dipole)
            )

            quadrupole_moment_data[atom_index] = CustomNamespace(
                q_xy=float(q_xy), q_xz=float(q_xz), q_yz=float(q_yz), q_x2_y2=float(q_x2_y2), q_3z2_r2=float(q_3z2_r2)
            )

        r_cubed_file_name = 'DDEC_atomic_Rcubed_moments.xyz'

        with open(r_cubed_file_name, 'r+') as vol_file:
            lines = vol_file.readlines()

        vols = [float(line.split()[-1]) for line in lines[2:atom_total + 2]]

        for atom_index in ddec_data:
            ddec_data[atom_index].volume = vols[atom_index]

        return ddec_data, dipole_moment_data, quadrupole_moment_data

    def _extract_charge_data_onetep(self):
        """
        From ONETEP output files, extract the necessary parameters for calculation of L-J.
        Insert data into ddec_data in standard format.
        Used exclusively by LennardJones class.
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

        charges = [float(line.split()[-1]) for line in lines[charge_pos: charge_pos + len(self.molecule.atoms)]]

        # Add the AIM-Valence and the AIM-Core to get V^AIM
        volumes = [float(line.split()[2]) + float(line.split()[3])
                   for line in lines[vol_pos: vol_pos + len(self.molecule.atoms)]]

        for atom_index in ddec_data:
            ddec_data[atom_index].charge = charges[atom_index]
            ddec_data[atom_index].volume = volumes[atom_index]

        return ddec_data


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
