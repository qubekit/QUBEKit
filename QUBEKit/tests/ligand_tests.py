from QUBEKit.ligand import Ligand

import unittest


class TestLigands(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.molecule = Ligand('tests/test_files/acetone.pdb')

    def test_pdb_reader(self):

        # Check all atoms are found
        assert(len(self.molecule.molecule) == 10)
        # Check atom names and coords are extracted for each atom in the molecule
        assert(len(i) == 4 for i in self.molecule.molecule)


if __name__ == '__main__':

    unittest.main()
