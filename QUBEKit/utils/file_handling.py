#!/usr/bin/env python3
"""
TODO
    Purpose of this file is to read various inputs and produce the info required for
        Ligand() or Protein()
        Currently, protein is not handled properly
    Should be very little calculation here, simply file reading and some small validations / checks
        Need to re-add topology checking and name checking
    Descriptors should be accessed separately if needed (need to re-add)
"""

from QUBEKit.engines import RDKit
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import Atom, Element
from QUBEKit.utils.exceptions import FileTypeError

from pathlib import Path
import re

import networkx as nx
import numpy as np


class ReadInput:
    def __init__(self, mol_input, name=None):

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

        self.read_input()

        # Try again for rdkit_mol

    def read_input(self):

        if self.mol_input.__class__.__name__ == 'Molecule':
            # QCArchive object
            self._read_qc_json()

        elif hasattr(self.mol_input, 'stem'):
            # File (pdb, xyz, etc)
            try:
                # Try parse with rdkit:
                self.rdkit_mol = RDKit.mol_input_to_rdkit_mol(self.mol_input, self.name)
                self._mol_from_rdkit()
            except AttributeError:
                # Cannot be parsed by rdkit:
                self._read_file()

        elif isinstance(self.mol_input, str):
            # Smiles string input
            self.rdkit_mol = RDKit.smiles_to_rdkit_mol(self.mol_input, self.name)
            self._mol_from_rdkit()

        else:
            raise RuntimeError('Cannot read input')

    def _read_file(self):

        if self.mol_input.suffix == '.pdb':
            self._read_pdb()
        elif self.mol_input.suffix == '.mol2':
            self._read_mol2()
        elif self.mol_input.suffix == '.xyz':
            self._read_xyz()
        else:
            raise FileTypeError(f'Could not read file {self.mol_input}')

    def _mol_from_rdkit(self):

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

        coords = []
        self.topology = nx.Graph()
        atoms = []

        atom_count = 0

        print('called!')
        with open(self.mol_input) as pdb:

            for line in pdb:
                if 'ATOM' in line or 'HETATM' in line:
                    print('reading!')
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
