from QUBEKit.utils.helpers import unpickle
from QUBEKit.mod_seminario import ModSeminario

import unittest
import os
import tempfile
from shutil import copy


class TestSeminario(unittest.TestCase):

    def setUp(self):
        """
        Set up the seminario test case by loading a pickled ligand that contains the hessian already
        :return: None
        """

        self.home = os.getcwd()
        self.test_folder = os.path.join(os.path.dirname(__file__), 'files')

        # Make temp folder and move the pickle file in
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            copy(os.path.join(self.test_folder, '.QUBEKit_states'), '.QUBEKit_states')
            self.molecules = unpickle()
            self.benzene_hessian, self.benzene_mod_sem = self.molecules['hessian'], self.molecules['mod_sem']
            self.benzene_mod_sem_vib_1 = self.molecules['mod_sem_vib_1']
            self.benzonitrile_hessian, self.benzonitrile_mod_sem = self.molecules['benzonitrile_hessian'], self.molecules['benzonitrile_mod_sem']

    def test_mod_sem(self):
        """Using wB97XD/6-311++G(d,p), scalling 0.957"""
        # Make temp folder and move the pickle file in
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            self.mod_sem = ModSeminario(self.benzene_hessian)

            # Get the seminario predicted values
            self.mod_sem.modified_seminario_method()
            self.assertEqual(self.benzene_hessian.HarmonicBondForce, self.benzene_mod_sem.HarmonicBondForce)
            self.assertEqual(self.benzene_hessian.HarmonicAngleForce, self.benzene_mod_sem.HarmonicAngleForce)

    def test_mod_sem_scalling(self):
        """Using wB97XD/6-311++G(d,p), scalling 1"""
        # Make temp folder and move the pickle file in
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            self.benzene_hessian.vib_scaling = 1
            self.mod_sem = ModSeminario(self.benzene_hessian)

            # Compare the seminario predicted values
            self.mod_sem.modified_seminario_method()
            self.assertEqual(self.benzene_hessian.HarmonicBondForce, self.benzene_mod_sem_vib_1.HarmonicBondForce)
            self.assertEqual(self.benzene_hessian.HarmonicAngleForce, self.benzene_mod_sem_vib_1.HarmonicAngleForce)

    def test_mod_sem_special_case(self):
        """Using xB97XD/6-311++G(d,p), scalling 0.957 on benzonitrile"""
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            self.mod_sem = ModSeminario(self.benzonitrile_hessian)

            # Compare the seminario predicted values
            self.mod_sem.modified_seminario_method()
            self.assertEqual(self.benzonitrile_hessian.HarmonicBondForce, self.benzonitrile_mod_sem.HarmonicBondForce)
            self.assertEqual(self.benzonitrile_hessian.HarmonicAngleForce, self.benzonitrile_mod_sem.HarmonicAngleForce)

    def tearDown(self):
        """Remove the temp folder"""
        os.chdir(self.home)


if __name__ == '__main__':

    unittest.main()
