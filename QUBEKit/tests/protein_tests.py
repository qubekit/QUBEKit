from QUBEKit.ligand import Protein

import os
from shutil import copy, rmtree
import unittest


class TestProteins(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.files_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
        os.mkdir('temp')
        os.chdir('temp')
        copy(os.path.join(cls.files_folder, 'capped_leu.pdb'), 'capped_leu.pdb')
        cls.molecule = Protein('capped_leu.pdb')

    def test_xml_generation(self):

        # Check all atoms are found in pdb file
        self.assertEqual(len(self.molecule.atoms), 31)

        # Check that every bond has been a length
        self.assertEqual(len(self.molecule.topology.edges), len(self.molecule.bond_lengths))

        # Check for angles and torsions too

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""
        os.chdir('../')
        rmtree('temp')


if __name__ == '__main__':

    unittest.main()
