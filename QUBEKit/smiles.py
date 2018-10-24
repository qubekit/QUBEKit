#!/usr/bin/env python


# Created on Thu Sep 20 12:55:13 2018

# @author: venkatakrishnan, edited by: Chris Ringrose


from rdkit.Chem import AllChem as Chem


def smiles_to_pdb(smiles_string):
    """Converts smiles strings to pdb and mol files"""

    if 'H' in str(smiles_string):
        raise Exception('Smiles string contains hydrogens, try again.')

    print('Creating pdb file from smiles string')

    m = Chem.MolFromSmiles(smiles_string)
    name = input('Please enter a name for the molecule:\n>')
    m.SetProp("_Name", name)
    mH = Chem.AddHs(m)
    Chem.EmbedMolecule(mH, Chem.ETKDG())
    Chem.SanitizeMol(mH)

    print(Chem.MolToMolBlock(mH), file=open('{}.mol'.format(name), 'w+'))
    Chem.MolToPDBFile(mH, "{}.pdb".format(name))
    print('Smiles string {} converted to PDB and mol files.'.format(smiles_string))
