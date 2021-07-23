"""
All tests are ran on ethane optimised with psi4 at b3lyp-d3bj/dzvp.
"""

import numpy as np

from qubekit.bonded import ModSeminario
from qubekit.molecules import Ligand
from qubekit.utils import constants
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
    The units returned from QM jobs was changed which caused an error in the Modified seminario method
    here we try and recreate a result before the unit rework.
    """
    with tmpdir.as_cwd():
        # load coords at the qm geometry
        mol = Ligand.from_file(get_data("chloromethane.sdf"))
        hessian = np.loadtxt(fname=get_data("chloromethane_hessian.txt"))
        mol.hessian = hessian
        mod_sem = ModSeminario()
        mod_sem.run(molecule=mol)

        # check the C-CL bond
        assert round(mol.BondForce[(0, 1)].length, ndigits=4) == 0.1805
        # get in kcal/mol like the reference values
        assert round(mol.BondForce[(0, 1)].k * constants.KJ_TO_KCAL / 200, 2) == 155.640
        # check a C-H bond
        assert round(mol.BondForce[(0, 2)].length, 3) == 0.109
        assert round(mol.BondForce[(0, 2)].k * constants.KJ_TO_KCAL / 200, 2) == 379.88

        # check a CL-C-H angle
        assert (
            round(mol.AngleForce[(1, 0, 2)].angle * constants.RAD_TO_DEG, 2) == 108.47
        )
        assert round(mol.AngleForce[(1, 0, 2)].k * constants.KJ_TO_KCAL / 2, 2) == 37.01

        # check a H-C-H angle
        assert (
            round(mol.AngleForce[(2, 0, 3)].angle * constants.RAD_TO_DEG, 2) == 110.45
        )
        assert round(mol.AngleForce[(2, 0, 3)].k * constants.KJ_TO_KCAL / 2, 2) == 37.49
