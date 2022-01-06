"""
All tests are ran on ethane optimised with psi4 at b3lyp-d3bj/dzvp.
"""

import numpy as np

from qubekit.bonded import ModSeminario
from qubekit.molecules import Ligand
from qubekit.utils.file_handling import get_data


def test_mod_sem_symmetry(tmpdir):
    """
    A simple regression test for modified seminario method with symmetry.
    """
    with tmpdir.as_cwd():
        # the sdf file is at the qm geometry
        mol = Ligand.from_file(file_name=get_data("ethane.sdf"))
        hessian = np.loadtxt(fname=get_data("ethane_hessian.txt"))
        mol.hessian = hessian
        mod_sem = ModSeminario()
        mod_sem.run(molecule=mol)
        # make sure symmetry groups are the same
        for bonds in mol.bond_types.values():
            values = set()
            for bond in bonds:
                values.add(tuple([mol.BondForce[bond].length, mol.BondForce[bond].k]))
            assert len(values) == 1
        for angles in mol.angle_types.values():
            values = set()
            for angle in angles:
                values.add(
                    tuple([mol.AngleForce[angle].angle, mol.AngleForce[angle].k])
                )
                assert len(values) == 1


def test_hessian_unit_regression(tmpdir):
    """
    The modsem method was found to have a bug in the scaling code which caused all angles to be scaled by the same amount.
    Here we try to reproduce some reference values for methanol which has a mix of scaled and non scaled angles.
    """
    with tmpdir.as_cwd():
        # load coords at the qm geometry
        mol = Ligand.parse_file(get_data("methanol.json"))

        mod_sem = ModSeminario()
        mod_sem.run(molecule=mol)

        # check the C-O bond
        assert round(mol.BondForce[(0, 1)].length, ndigits=4) == 0.1413
        # get in kcal/mol like the reference values
        assert round(mol.BondForce[(0, 1)].k, 3) == 246439.036
        # check a O-H bond
        assert round(mol.BondForce[(1, 5)].length, 4) == 0.0957
        assert round(mol.BondForce[(1, 5)].k, 2) == 513819.18

        # check the C-O-H angle
        assert round(mol.AngleForce[(0, 1, 5)].angle, 3) == 1.899
        assert round(mol.AngleForce[(0, 1, 5)].k, 3) == 578.503

        # check a scaled H-C-H angle
        assert round(mol.AngleForce[(2, 0, 3)].angle, 3) == 1.894
        assert round(mol.AngleForce[(2, 0, 3)].k, 3) == 357.05
