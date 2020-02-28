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

from QUBEKit.engines import Element, RDKit
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import Atom
from QUBEKit.utils.exceptions import FileTypeError

import re

import networkx as nx
import numpy as np


def read_input(mol_input, name=None):
    """
    Main access point. Loading initial data into QUBEKit should come through here
    This function is therefore what is called by Ligand's init
    :param mol_input: Can be a Path, smiles string or qc_json
    :param name: name of molecule
    :return: name, topology, atoms and coords of molecule
    """

    if mol_input.__class__.__name__ == 'Molecule':
        topology, atoms, coords = _read_qc_json(mol_input)

    # TODO cleaner way of doing this?
    elif hasattr(mol_input, 'stem'):
        try:
            rdkit_mol = RDKit.read_file(mol_input)
            name, topology, atoms, coords = _mol_from_rdkit(rdkit_mol, name)
        except AttributeError:
            topology, atoms, coords = _read_file(mol_input)

    elif isinstance(mol_input, str):
        rdkit_mol = RDKit.smiles_to_rdkit_mol(mol_input, name)
        name, topology, atoms, coords = _mol_from_rdkit(rdkit_mol, name)

    else:
        raise RuntimeError('Cannot read input')

    return name, topology, atoms, coords


def _read_file(file_path):

    if file_path.suffix == '.pdb':
        return _read_pdb(file_path)
    elif file_path.suffix == '.mol2':
        return _read_mol2(file_path)
    elif file_path.suffix == '.xyz':
        return None, None, _read_xyz(file_path)
    else:
        raise FileTypeError(f'Could not read file {file_path}')


def _mol_from_rdkit(rdkit_mol, name=None):

    if name is None:
        name = rdkit_mol.GetProp('_Name')

    atoms = []
    topology = nx.Graph()
    # Collect the atom names and bonds
    for atom in rdkit_mol.GetAtoms():
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
        topology.add_node(atom.GetIdx())

        # Add the bonds
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())
            qube_atom.add_bond(bonded.GetIdx())

        # Now add the atom to the molecule
        atoms.append(qube_atom)

    # Now get the coordinates
    coords = rdkit_mol.GetConformer().GetPositions()

    return name, topology, atoms, coords


def _read_qc_json(mol_input):

    topology = nx.Graph()
    atoms = []

    for i, atom in enumerate(mol_input.symbols):
        atoms.append(Atom(atomic_number=Element().number(atom), atom_index=i, atom_name=f'{atom}{i}'))
        topology.add_node(i)

    for bond in mol_input.connectivity:
        topology.add_edge(*bond[:2])

    coords = np.array(mol_input.geometry).reshape((len(atoms), 3)) * constants.BOHR_TO_ANGS

    return topology, atoms, coords


def _read_pdb(filename):

    coords = []
    topology = nx.Graph()
    atoms = []

    atom_count = 0

    with open(filename) as pdb:

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
                topology.add_node(atom_count)
                atom_count += 1
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

            if 'CONECT' in line:
                atom_index = int(line.split()[1]) - 1
                # Search the connectivity section and add all edges to the graph corresponding to the bonds.
                for i in range(2, len(line.split())):
                    if int(line.split()[i]) != 0:
                        bonded_index = int(line.split()[i]) - 1
                        topology.add_edge(atom_index, bonded_index)
                        atoms[atom_index].add_bond(bonded_index)
                        atoms[bonded_index].add_bond(atom_index)

    # put the object back into the correct place
    coords = np.array(coords)

    return topology, atoms, coords


def _read_mol2(input_file):

    molecule = []
    topology = nx.Graph()
    atoms = []

    atom_count = 0

    with open(input_file, 'r') as mol2:

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

                molecule.append([float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])

                # Collect the atom names
                atom_name = str(line.split()[1])

                # Add the nodes to the topology object
                topology.add_node(atom_count)
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
                topology.add_edge(atom_index, bonded_index)
                atoms[atom_index].add_bond(bonded_index)
                atoms[bonded_index].add_bond(atom_index)

    # put the object back into the correct place
    coords = np.array(molecule)

    return topology, atoms, coords


def _read_xyz(filename):

    traj_molecules = []
    molecule = []

    with open(filename) as xyz_file:
        lines = xyz_file.readlines()

        n_atoms = float(lines[0])

        for line in lines:
            line = line.split()
            # skip frame heading lines
            if len(line) <= 1 or 'Iteration' in line:
                continue

            molecule.append([float(line[1]), float(line[2]), float(line[3])])

            if len(molecule) == n_atoms:
                # we have collected the molecule now store the frame
                traj_molecules.append(np.array(molecule))
                molecule = []

    return traj_molecules[0] if len(traj_molecules) == 1 else traj_molecules
