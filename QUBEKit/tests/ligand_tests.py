from QUBEKit.ligand import Ligand

import unittest


class TestLigands(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.molecule = Ligand('test_files/acetone.pdb')

    def test_pdb_reader(self):

        # Check all atoms are found
        self.assertEqual(10, len(self.molecule.molecule['input']))

        # Check atom names and coords are extracted for each atom in the molecule
        for atom in self.molecule.molecule['input']:
            self.assertEqual(4, len(atom))


if __name__ == '__main__':

    unittest.main()
