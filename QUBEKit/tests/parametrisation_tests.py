from QUBEKit.parametrisation import AnteChamber, OpenFF
from QUBEKit.ligand import Ligand
from QUBEKit.tests.test_structures import acetone

import os
import unittest


class ParametrisationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Write the big string above to a file to be used for testing.
        Cannot use actual files as pathing causes issues.
        """
        with open('acetone.pdb', 'w+') as pdb_test_file:
            pdb_test_file.write(acetone)

        cls.molecule = Ligand('acetone.pdb')

    def test_antechamber(self):
        # try to parametrise the molecule with antechamber
        AnteChamber(self.molecule)

        # now make sure we have parameters assigned
        self.assertEqual(len(self.molecule.HarmonicBondForce), len(list(self.molecule.topology.edges)))

        self.assertEqual(len(self.molecule.HarmonicAngleForce), len(self.molecule.angles))

        self.assertEqual(len(self.molecule.PeriodicTorsionForce),
                         len(self.molecule.dih_phis) + len(self.molecule.improper_torsions))

        self.assertEqual(len(self.molecule.coords['input']), len(self.molecule.NonbondedForce))

    def test_OpenFF(self):
        # try to parametrise using OpenFF
        OpenFF(self.molecule)

        # now make sure we have parameters assigned
        self.assertEqual(len(self.molecule.HarmonicBondForce), len(list(self.molecule.topology.edges)))

        self.assertEqual(len(self.molecule.HarmonicAngleForce), len(self.molecule.angles))

        self.assertEqual(len(self.molecule.PeriodicTorsionForce),
                         len(self.molecule.dih_phis) + len(self.molecule.improper_torsions))

        self.assertEqual(len(self.molecule.coords['input']), len(self.molecule.NonbondedForce))

    @classmethod
    def tearDownClass(cls):
        """Remove the files produced during testing"""
        os.system('rm *.frcmod *.inpcrd *.mol2 *.prmtop *.log serialised.xml acetone.pdb')


if __name__ == '__main__':

    unittest.main()
