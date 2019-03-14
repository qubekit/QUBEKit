from QUBEKit.ligand import Protein

import unittest


class TestProteins(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.molecule = Protein('tests/test_files/aceleunme.pdb')

    def test_xml_generation(self):

        self.molecule.atom_types()
        self.molecule.construct_xml_dicts()

        # Check all atoms are found in pdb file
        assert(len(self.molecule.molecule) == 31)

        # Check that each bond has an associated HarmonicBondForce
        self.assertEqual(sorted(self.molecule.bonds), sorted(self.molecule.HarmonicBondForce))

        # Check for angles and torsions too
        #
        #


if __name__ == '__main__':

    unittest.main()
