#Smiles to pdb and mol files.
"""
Created on Thu Sep 20 12:55:13 2018

@author: venkatakrishnan
"""
from rdkit.Chem import AllChem as Chem
import sys
def smiles_to_pdb(smiles_string):
    if 'H' in str(smiles_string):
        sys.exit('smiles string contains hydrogens try again!')
    print('creating pdb file from smiles string')
    m = Chem.MolFromSmiles('%s'%smiles_string)
    name = input('please enter a name for the molecule\n>')
    m.SetProp("_Name", name)
    mH = Chem.AddHs(m)
    Chem.EmbedMolecule(mH,Chem.ETKDG())
    Chem.SanitizeMol(mH)
    print(Chem.MolToMolBlock(mH),file=open('%s.mol'%(name),'w+'))
    Chem.MolToPDBFile(mH, "%s.pdb"%(name))
    print('smiles string %s converted to PDB and mol files!'%(smiles_string))
    
