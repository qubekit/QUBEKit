#!/usr/bin/env python

from rdkit.Chem import AllChem, MolFromPDBFile, Descriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule


def smiles_to_pdb(smiles_string, name=None):
    """Converts smiles strings to pdb and mol files."""
    # Originally written by: venkatakrishnan, rewritten and extended by: Chris Ringrose

    if 'H' in smiles_string:
        raise SyntaxError('Smiles string contains hydrogen atoms; try again.')

    print('Creating pdb file from smiles string.')

    m = AllChem.MolFromSmiles(smiles_string)
    if not name:
        name = input('Please enter a name for the molecule:\n>')
    m.SetProp('_Name', name)
    mH = AllChem.AddHs(m)
    AllChem.EmbedMolecule(mH, AllChem.ETKDG())
    AllChem.SanitizeMol(mH)

    print(AllChem.MolToMolBlock(mH), file=open(f'{name}.mol', 'w+'))
    AllChem.MolToPDBFile(mH, f'{name}.pdb')
    print(f'Smiles string {smiles_string} converted to PDB and mol files.')

    return f'{name}.pdb'


def smiles_mm_optimise(pdb_file, ff='MMF'):
    """
    Perform rough preliminary optimisation to speed up later optimisations
    and extract some extra information about the molecule.
    """

    force_fields = {'MMF': MMFFOptimizeMolecule, 'UFF': UFFOptimizeMolecule}

    mol = MolFromPDBFile(pdb_file, removeHs=False)

    force_fields[ff](mol)

    AllChem.MolToPDBFile(mol, f'{pdb_file[:-4]}_rdkit_optimised.pdb')

    return f'{pdb_file[:-4]}_rdkit_optimised.pdb'


def rdkit_descriptors(pdb_file):
    """
    Use RDKit Descriptors to extract properties and store in Descriptors dictionary
    :param pdb_file: The molecule input file
    :return: descriptors dictionary
    """

    mol = MolFromPDBFile(pdb_file, removeHs=False)
    # Use RDKit Descriptors to extract properties and store in Descriptors dictionary
    descriptors = {'Heavy atoms': Descriptors.HeavyAtomCount(mol),
                   'H-bond donors': Descriptors.NumHDonors(mol),
                   'H-bond acceptors': Descriptors.NumHAcceptors(mol),
                   'Molecular weight': Descriptors.MolWt(mol),
                   'LogP': Descriptors.MolLogP(mol)}

    return descriptors
