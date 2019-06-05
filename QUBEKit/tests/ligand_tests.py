from QUBEKit.ligand import Ligand
from QUBEKit.tests.test_structures import acetone, acetone_mol2

import unittest
from os import system


class TestLigands(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Write the big string in test_structures to a file to be used for testing.
        Cannot use actual files as pathing causes issues.
        """
        with open('acetone.pdb', 'w+') as pdb_test_file:
            pdb_test_file.write(acetone)

        with open('acetone.mol2', 'w+') as mol2_test_file:
            mol2_test_file.write(acetone_mol2)

        cls.molecule_pdb = Ligand('acetone.pdb')
        cls.molecule_mol2 = Ligand('acetone.mol2')

    def test_pdb_reader(self):
        # Make sure the pdb reader has been used

        # Check all atoms are found
        self.assertEqual(10, len(self.molecule_pdb.atoms))

        # Check atom names and coords are extracted for each atom in the molecule
        for atom in self.molecule_pdb.molecule['input']:
            self.assertEqual(3, len(atom))

    def test_mol2_reader(self):
        # Make sure the mol2 reader has been used

        # Check all atoms are found
        self.assertEqual(10, len(self.molecule_mol2.atoms))

        # Check atom names and coords are extracted for each atom in the molecule
        for atom in self.molecule_mol2.molecule['input']:
            self.assertEqual(3, len(atom))

    def test_bonds(self):

        # check we have found the bonds in the connections table
        bonds = [(0, 4), (0, 5), (0, 6), (0, 1), (1, 2), (1, 3), (3, 7), (3, 8), (3, 9)]
        self.assertEqual(bonds, list(self.molecule_pdb.topology.edges))
        self.assertEqual(bonds, list(self.molecule_mol2.topology.edges))

        # make sure every bond has a length
        self.assertEqual(len(self.molecule_pdb.bond_lengths), len(list(self.molecule_pdb.topology.edges)))
        self.assertEqual(len(self.molecule_mol2.bond_lengths), len(list(self.molecule_mol2.topology.edges)))

    def test_angles(self):

        # check that we have found all angles in the molecule
        angles = [(1, 0, 4), (1, 0, 5), (1, 0, 6), (4, 0, 5), (4, 0, 6), (5, 0, 6), (0, 1, 2), (0, 1, 3), (2, 1, 3),
                  (1, 3, 7), (1, 3, 8), (1, 3, 9), (7, 3, 8), (7, 3, 9), (8, 3, 9)]

        self.assertEqual(angles, list(self.molecule_pdb.angles))
        self.assertEqual(angles, list(self.molecule_mol2.angles))

        # make sure every angle has a value
        self.assertEqual(len(self.molecule_pdb.angles), len(self.molecule_pdb.angle_values))
        self.assertEqual(len(self.molecule_mol2.angles), len(self.molecule_mol2.angle_values))

    def test_dihedrals(self):

        # check the dihedral angles in the molecule
        dihedrals = {(0, 1): [(4, 0, 1, 2), (4, 0, 1, 3), (5, 0, 1, 2), (5, 0, 1, 3), (6, 0, 1, 2), (6, 0, 1, 3)],
                     (1, 3): [(0, 1, 3, 7), (0, 1, 3, 8), (0, 1, 3, 9), (2, 1, 3, 7), (2, 1, 3, 8), (2, 1, 3, 9)]}
        self.assertEqual(dihedrals, self.molecule_pdb.dihedrals)
        self.assertEqual(dihedrals, self.molecule_mol2.dihedrals)

        # check that every dihedral has a value measured
        self.assertEqual(12, len(self.molecule_pdb.dih_phis))
        self.assertEqual(12, len(self.molecule_mol2.dih_phis))

        # check the improper dihedrals found
        impropers = [(1, 0, 2, 3)]
        self.assertEqual(impropers, self.molecule_pdb.improper_torsions)
        self.assertEqual(impropers, self.molecule_mol2.improper_torsions)

        # check the rotatable torsions found this ensures that the methyl groups are removed from the
        # torsion list and symmetry is working
        rot = []
        self.assertEqual(rot, self.molecule_pdb.rotatable)
        self.assertEqual(rot, self.molecule_mol2.rotatable)

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""
        system('rm acetone.pdb acetone.mol2')


if __name__ == '__main__':

    unittest.main()
