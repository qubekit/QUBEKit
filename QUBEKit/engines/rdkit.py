#!/usr/bin/env python3

from QUBEKit.decorators import for_all_methods, timer_logger

from pathlib import Path

from rdkit.Chem import AllChem, MolFromPDBFile, Descriptors, MolToSmiles, MolToSmarts, MolToMolFile, MolFromMol2File, MolFromMolFile, rdPartialCharges
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule


@for_all_methods(timer_logger)
class RDKit:
    """Class for controlling useful RDKit functions; try to keep class static."""

    def __init__(self):
        pass

    def read_file(self, filename):

        # This handles splitting the paths
        filename = Path(filename)

        # Try and read the file
        if filename.suffix == '.pdb':
            mol = MolFromPDBFile(filename.name, removeHs=False)
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except RuntimeError:
                print('RDKit could not assign the partial charges')
        elif filename.suffix == '.mol2':
            mol = MolFromMol2File(filename.name, removeHs=False)
        elif filename.suffix == '.mol':
            mol = MolFromMolFile(filename.name, removeHs=False)
        else:
            mol = None

        return mol

    def smiles_to_rdkit_mol(self, smiles_string, name=None):
        """
        Converts smiles strings to RDKit molobject.
        :param smiles_string: The hydrogen free smiles string
        :param name: The name of the molecule this will be used when writing the pdb file
        :return: The RDKit molecule
        """
        # Originally written by venkatakrishnan; rewritten and extended by Chris Ringrose

        if 'H' in smiles_string:
            raise SyntaxError('Smiles string contains hydrogen atoms; try again.')

        m = AllChem.MolFromSmiles(smiles_string)
        if name is None:
            name = input('Please enter a name for the molecule:\n>')
        m.SetProp('_Name', name)
        mol_hydrogens = AllChem.AddHs(m)
        AllChem.EmbedMolecule(mol_hydrogens, AllChem.ETKDG())
        AllChem.SanitizeMol(mol_hydrogens)
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol_hydrogens)
        except RuntimeError:
            print('RDKit could not assign the partial charges')

        #print(AllChem.MolToMolBlock(mol_hydrogens), file=open(f'{name}.mol', 'w+'))
        #AllChem.MolToPDBFile(mol_hydrogens, f'{name}.pdb')

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

        force_fields = {'MMF': MMFFOptimizeMolecule, 'UFF': UFFOptimizeMolecule}

        force_fields[ff](mol)

        AllChem.MolToPDBFile(mol, f'{filename.stem}_rdkit_optimised.pdb')

        return f'{filename.stem}_rdkit_optimised.pdb'

    def rdkit_descriptors(self, filename):
        """
        Use RDKit Descriptors to extract properties and store in Descriptors dictionary.
        :param filename: The molecule input file
        :return: Descriptors dictionary
        """

        mol = RDKit().read_file(filename)
        # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
        descriptors = {'Heavy atoms': Descriptors.HeavyAtomCount(mol),
                       'H-bond donors': Descriptors.NumHDonors(mol),
                       'H-bond acceptors': Descriptors.NumHAcceptors(mol),
                       'Molecular weight': Descriptors.MolWt(mol),
                       'LogP': Descriptors.MolLogP(mol)}

        return descriptors

    def get_smiles(self, filename):
        """
        Use RDKit to load in the pdb file of the molecule and get the smiles code.
        :param filename: The molecule input file
        :return: The smiles string
        """

        mol = RDKit().read_file(filename)

        return MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)

    def get_smarts(self, filename):
        """
        Use RDKit to get the smarts string of the molecule.
        :param filename: The molecule input file
        :return: The smarts string
        """

        mol = RDKit().read_file(filename)

        return MolToSmarts(mol)

    def get_mol(self, filename):
        """
        Use RDKit to generate a mol file.
        :param filename: The molecule input file
        :return: The name of the mol file made
        """

        mol = RDKit().read_file(filename)

        mol_name = f'{filename.steam}.mol'
        MolToMolFile(mol, mol_name)

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
        coords = [conformer.GetPositions() for conformer in positions]

        return coords
