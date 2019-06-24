from QUBEKit.ligand import Protein
from QUBEKit.tests.test_structures import aceleunme

import os
import unittest


class TestProteins(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Write the big string in test_structures to a file to be used for testing.
        Cannot use actual files as pathing causes issues.
        """
        with open('aceleunme.pdb', 'w+') as pdb_test_file:
            pdb_test_file.write(aceleunme)

        cls.molecule = Protein('aceleunme.pdb')

    def test_xml_generation(self):

        self.molecule.AtomTypes()

        # Check all atoms are found in pdb file
        self.assertEqual(len(self.molecule.coords), 31)

        # Check that each bond has an associated HarmonicBondForce
        self.assertEqual(sorted(self.molecule.bond_lengths), sorted(self.molecule.HarmonicBondForce))

        # Check for angles and torsions too

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""
        os.system('rm aceleunme.pdb')


if __name__ == '__main__':

    unittest.main()
