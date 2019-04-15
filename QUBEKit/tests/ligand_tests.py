from QUBEKit.ligand import Ligand
from QUBEKit.tests.test_structures import acetone

from os import system

import unittest


class TestLigands(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Write the big string above to a file to be used for testing.
        Cannot use actual files as pathing causes issues.
        """
        with open('acetone.pdb', 'w+') as pdb_test_file:
            pdb_test_file.write(acetone)

        cls.molecule = Ligand('acetone.pdb')

    def test_pdb_reader(self):

        # Check all atoms are found
        self.assertEqual(10, len(self.molecule.molecule['input']))

        # Check atom names and coords are extracted for each atom in the molecule
        for atom in self.molecule.molecule['input']:
            self.assertEqual(4, len(atom))

    def test_bonds(self):

        # check we have found the bonds in the conectons table
        bonds = [(1, 5), (1, 6), (1, 7), (1, 2), (2, 3), (2, 4), (4, 8), (4, 9), (4, 10)]
        self.assertEqual(bonds, list(self.molecule.topology.edges))

        # make sure every bond has a length
        self.assertEqual(len(self.molecule.bond_lengths), len(list(self.molecule.topology.edges)))

    def test_angles(self):

        # check that we have found all angles in the molecule
        angles = [(2, 1, 5), (2, 1, 6), (2, 1, 7), (5, 1, 6), (5, 1, 7), (6, 1, 7), (1, 2, 3), (1, 2, 4),
                  (3, 2, 4), (2, 4, 8), (2, 4, 9), (2, 4, 10), (8, 4, 9), (8, 4, 10), (9, 4, 10)]

        self.assertEqual(angles, list(self.molecule.angles))

        # make sure every angle has a value
        self.assertEqual(len(self.molecule.angles), len(self.molecule.angle_values))

    def test_dihedrals(self):

        # check the dihedral angles in the molecule
        dihedrals = {(1, 2): [(5, 1, 2, 3), (5, 1, 2, 4), (6, 1, 2, 3), (6, 1, 2, 4), (7, 1, 2, 3), (7, 1, 2, 4)],
                     (2, 4): [(1, 2, 4, 8), (1, 2, 4, 9), (1, 2, 4, 10), (3, 2, 4, 8), (3, 2, 4, 9), (3, 2, 4, 10)]}
        self.assertEqual(dihedrals, self.molecule.dihedrals)

        # check that every dihedral has a value measured
        self.assertEqual(12, len(self.molecule.dih_phis))

        # check the improper dihedrals found
        impropers = [(2, 1, 3, 4)]
        self.assertEqual(impropers, self.molecule.improper_torsions)

        # check the rotatable torsions found this ensures that the methyl groups are removed from the
        # torsion list and symmetry is working
        rot = []
        self.assertEqual(rot, self.molecule.rotatable)

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""
        system('rm acetone.pdb')


if __name__ == '__main__':

    unittest.main()
