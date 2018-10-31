#!/usr/bin/env python


# Created on Thu Sep 20 12:55:13 2018

# @author: venkatakrishnan, edited by: Chris Ringrose


from rdkit.Chem import AllChem as Chem
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeForceField, MMFFGetMoleculeProperties, MMFFOptimizeMolecule
from rdkit.Chem.AllChem import EmbedMolecule


def smiles_to_pdb(smiles_string):
    """Converts smiles strings to pdb and mol files"""

    if 'H' in str(smiles_string):
        raise Exception('Smiles string contains hydrogen atoms; try again.')

    print('Creating pdb file from smiles string.')

    m = Chem.MolFromSmiles(smiles_string)
    name = input('Please enter a name for the molecule:\n>')
    m.SetProp('_Name', name)
    mH = Chem.AddHs(m)
    Chem.EmbedMolecule(mH, Chem.ETKDG())
    Chem.SanitizeMol(mH)

    print(Chem.MolToMolBlock(mH), file=open(f'{name}.mol', 'w+'))
    Chem.MolToPDBFile(mH, f'{name}.pdb')
    print(f'Smiles string {smiles_string} converted to PDB and mol files.')
    return name + '.pdb'


def smiles_mm_optimise(pdb_file):
    """Perform rough preliminary optimisation to speed up later optimisations."""
    mol = MolFromPDBFile(pdb_file, removeHs=False)
    EmbedMolecule(mol)
    # mol_properties = MMFFGetMoleculeProperties(mol)
    # ff = MMFFGetMoleculeForceField(mol, mol_properties)
    MMFFOptimizeMolecule(mol)
    Chem.MolToPDBFile(mol, pdb_file[:-4] + '_rdkit_optimised.pdb')
    return pdb_file[:-4] + '_rdkit_optimised.pdb'
