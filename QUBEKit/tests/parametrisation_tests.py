from QUBEKit.parametrisation import AnteChamber, OpenFF
from QUBEKit.ligand import Ligand


import os
import unittest
from shutil import copy, rmtree


class ParametrisationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.files_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')

        # Make the temp folder and move there with the required files
        os.mkdir('temp')
        os.chdir('temp')
        copy(os.path.join(cls.files_folder, 'acetone.pdb'), 'acetone.pdb')
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
        os.chdir('../')
        rmtree('temp')


if __name__ == '__main__':

    unittest.main()
