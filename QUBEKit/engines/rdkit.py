#!/usr/bin/env python3

from QUBEKit.utils.decorators import for_all_methods, timer_logger

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import GetPeriodicTable
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule


@for_all_methods(timer_logger)
class RDKit:
    """Class for controlling useful RDKit functions."""
    def __init__(self):
        pass

    def read_file(self, filename):

        # Try and read the file
        if filename.suffix == '.pdb':
            mol = Chem.MolFromPDBFile(filename.name, removeHs=False)
            try:
                Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
            except RuntimeError:
                print('RDKit could not assign the partial charges')
        elif filename.suffix == '.mol2':
            mol = Chem.MolFromMol2File(filename.name, removeHs=False)
        elif filename.suffix == '.mol':
            mol = Chem.MolFromMolFile(filename.name, removeHs=False)
            try:
                Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
            except RuntimeError:
                print('RDKit could not assign the partial charges')
        else:
            mol = None

        return mol

    def smiles_to_rdkit_mol(self, smiles_string, name=None):
        """
        Converts smiles strings to RDKit mol object.
        :param smiles_string: The hydrogen free smiles string
        :param name: The name of the molecule this will be used when writing the pdb file
        :return: The RDKit molecule
        """

        mol = AllChem.MolFromSmiles(smiles_string)
        if name is None:
            name = input('Please enter a name for the molecule:\n>')
        mol.SetProp('_Name', name)
        mol_hydrogens = AllChem.AddHs(mol)
        AllChem.EmbedMolecule(mol_hydrogens, AllChem.ETKDG())
        AllChem.SanitizeMol(mol_hydrogens)
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol_hydrogens)
        except RuntimeError:
            print('RDKit could not assign the partial charges')

        return mol_hydrogens

    def mm_optimise(self, filename, ff='MMF'):
        """
        Perform rough preliminary optimisation to speed up later optimisations.
        :param filename: The name of the input file
        :param ff: The Force field to be used either MMF or UFF
        :return: The name of the optimised pdb file that is made
        """

        # Get the rdkit molecule
        mol = RDKit().read_file(filename)

        {'MMF': MMFFOptimizeMolecule, 'UFF': UFFOptimizeMolecule}[ff](mol)

        AllChem.MolToPDBFile(mol, f'{filename.stem}_rdkit_optimised.pdb')

        return f'{filename.stem}_rdkit_optimised.pdb'

    def rdkit_descriptors(self, rdkit_mol):
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary.
        :param rdkit_mol: The molecule input file
        :return: descriptors dictionary
        """

        # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        return {'Heavy atoms': Descriptors.HeavyAtomCount(rdkit_mol),
                'H-bond donors': Descriptors.NumHDonors(rdkit_mol),
                'H-bond acceptors': Descriptors.NumHAcceptors(rdkit_mol),
                'Molecular weight': Descriptors.MolWt(rdkit_mol),
                'LogP': Descriptors.MolLogP(rdkit_mol)}

    def get_smiles(self, filename):
        """
        Use RDKit to load in the pdb file of the molecule and get the smiles code.
        :param filename: The molecule input file
        :return: The smiles string
        """

        mol = RDKit().read_file(filename)

        return Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)

    def get_smarts(self, filename):
        """
        Use RDKit to get the smarts string of the molecule.
        :param filename: The molecule input file
        :return: The smarts string
        """

        mol = RDKit().read_file(filename)

        return Chem.MolToSmarts(mol)

    def get_mol(self, filename):
        """
        Use RDKit to generate a mol file.
        :param filename: The molecule input file
        :return: The name of the mol file made
        """

        mol = RDKit().read_file(filename)

        mol_name = f'{filename.stem}.mol'
        Chem.MolToMolFile(mol, mol_name)

        return mol_name

    def generate_conformers(self, filename, conformer_no=10):
        """
        Generate a set of x conformers of the molecule
        :param conformer_no: The amount of conformers made for the molecule
        :param filename: The name of the input file
        :return: A list of conformer position arrays
        """

        mol = RDKit().read_file(filename)

        cons = AllChem.EmbedMultipleConfs(mol, numConfs=conformer_no)
        positions = cons.GetConformers()

        return [conformer.GetPositions() for conformer in positions]

    def find_symmetry_classes(self, mol):
        """
        Generate list of tuples of symmetry-equivalent (homotopic) atoms in the molecular graph
        based on https://sourceforge.net/p/rdkit/mailman/message/27897393/
        :param mol: molecule to find symmetry classes for
        :return: A list of tuples of equivalent atom indices (zero-based)
        """

        # Check CIPRank is present for first atom (can assume it is present for all afterwards)
        if not mol.GetAtomWithIdx(0).HasProp('_CIPRank'):
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

        # Array of ranks showing matching atoms
        cip_ranks = np.array([int(atom.GetProp('_CIPRank')) for atom in mol.GetAtoms()])

        # Map the ranks to the atoms to produce a list of symmetrical atoms
        atom_symmetry_classes = [np.where(cip_ranks == rank)[0].tolist() for rank in range(max(cip_ranks) + 1)]

        # Convert from list of classes to dict where each key is an atom and each value is its class (just a str)
        atom_symmetry_classes_dict = {}
        # i will be used to define the class (just index based)
        for i, klass in enumerate(atom_symmetry_classes):
            for atom in klass:
                atom_symmetry_classes_dict[atom] = str(i)

        return atom_symmetry_classes_dict


class Element:
    """
    Simple wrapper class for getting element info using RDKit.
    """

    pt = GetPeriodicTable()

    def mass(self, identifier):
        return self.pt.GetAtomicWeight(identifier)

    def number(self, identifier):
        return self.pt.GetAtomicNumber(identifier)

    def name(self, identifier):
        return self.pt.GetElementSymbol(identifier)
