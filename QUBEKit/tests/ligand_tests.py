from QUBEKit.ligand import Ligand

import unittest


acetone = """COMPND    acetone
HETATM    1  C1  UNL     1       1.273  -0.213  -0.092  1.00  0.00           C  
HETATM    2  C2  UNL     1      -0.006   0.543  -0.142  1.00  0.00           C  
HETATM    3  O1  UNL     1       0.076   1.721  -0.474  1.00  0.00           O  
HETATM    4  C3  UNL     1      -1.283  -0.126   0.197  1.00  0.00           C  
HETATM    5  H1  UNL     1       1.797   0.136   0.840  1.00  0.00           H  
HETATM    6  H2  UNL     1       1.845   0.072  -1.000  1.00  0.00           H  
HETATM    7  H3  UNL     1       1.082  -1.311  -0.045  1.00  0.00           H  
HETATM    8  H4  UNL     1      -1.979   0.089  -0.663  1.00  0.00           H  
HETATM    9  H5  UNL     1      -1.680   0.295   1.147  1.00  0.00           H  
HETATM   10  H6  UNL     1      -1.125  -1.207   0.231  1.00  0.00           H  
CONECT    1    2    5    6    7
CONECT    2    3    3    4
CONECT    4    8    9   10
END
"""


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


if __name__ == '__main__':

    unittest.main()
