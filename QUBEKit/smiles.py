#!/usr/bin/env python


from rdkit.Chem import AllChem, MolFromPDBFile, Descriptors
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeForceField, MMFFGetMoleculeProperties, MMFFOptimizeMolecule


def smiles_to_pdb(smiles_string, name=None):
    """Converts smiles strings to pdb and mol files"""
    # Originally written by: venkatakrishnan, rewritten and extended by: Chris Ringrose

    if 'H' in str(smiles_string):
        raise Exception('Smiles string contains hydrogen atoms; try again.')

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


def smiles_mm_optimise(pdb_file):
    """Perform rough preliminary optimisation to speed up later optimisations
    and extract some extra infomation about the molecule."""

    mol = MolFromPDBFile(pdb_file, removeHs=False)
    AllChem.EmbedMolecule(mol)
    # use rdkit Descriptors to extract properties and store in Descriptors dictionary
    descriptors = {'Heavy atoms': Descriptors.HeavyAtomCount(mol),
                   'H-bond donors': Descriptors.NumHDonors(mol),
                   'H-bond acceptors': Descriptors.NumHAcceptors(mol),
                   'Molecular weight': Descriptors.MolWt(mol),
                   'LogP': Descriptors.MolLogP(mol)}
    # mol_properties = MMFFGetMoleculeProperties(mol)
    # ff = MMFFGetMoleculeForceField(mol, mol_properties)
    MMFFOptimizeMolecule(mol)
    AllChem.MolToPDBFile(mol, f'{pdb_file[:-4]}_rdkit_optimised.pdb')

    return f'{pdb_file[:-4]}_rdkit_optimised.pdb', descriptors
