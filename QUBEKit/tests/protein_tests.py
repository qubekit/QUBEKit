from QUBEKit.ligand import Protein

import unittest
import os
import tempfile
from shutil import copy


class TestProteins(unittest.TestCase):

    def setUp(self):
        """
        Set up a protein testing class, make temp folder and copy capped leu file
        """

        self.home = os.getcwd()
        self.test_folder = os.path.join(os.path.dirname(__file__), 'files')

        # Make the temp folder and move there with the required files
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            copy(os.path.join(self.test_folder, 'capped_leu.pdb'), 'capped_leu.pdb')
            self.molecule = Protein('capped_leu.pdb')

    def test_xml_generation(self):

        # Check all atoms are found in pdb file
        self.assertEqual(len(self.molecule.atoms), 31)

        # Check that every bond has been a length
        self.assertEqual(len(self.molecule.topology.edges), len(self.molecule.bond_lengths))

        # Check for angles and torsions too

    def tearDown(self):
        """Remove the files produced during testing"""
        os.chdir(self.home)


if __name__ == '__main__':

    unittest.main()
