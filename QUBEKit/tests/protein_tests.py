#!/usr/bin/env python3

import os
import unittest
from shutil import copy, rmtree

from QUBEKit.ligand import Protein


class TestProteins(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create temp working directory and copy across test files."""

        cls.files_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "files"
        )
        os.mkdir("temp")
        os.chdir("temp")
        copy(os.path.join(cls.files_folder, "capped_leu.pdb"), "capped_leu.pdb")
        cls.molecule = Protein("capped_leu.pdb")
        cls.molecule.testing = True

    def test_xml_generation(self):
        """Ensure atoms and bonds have been populated in Protein object."""

        self.assertEqual(len(self.molecule.atoms), 31)

        self.assertEqual(
            len(list(self.molecule.topology.edges)), len(self.molecule.bond_lengths)
        )

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""

        os.chdir("../")
        rmtree("temp")


if __name__ == "__main__":

    unittest.main()
