"""
All tests are ran on ethane optimised with psi4 at b3lyp-d3bj/dzvp.
"""

import numpy as np
import pytest

from QUBEKit.ligand import Ligand
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.utils.file_handling import get_data


def test_mod_sem_no_symmetry(tmpdir):
    """
    A simple regression test for the modified seminario method.
    # TODO expand these tests
    """
    with tmpdir.as_cwd():
        # the sdf file is at the qm geometry
        mol = Ligand.from_file(file_name=get_data("ethane.sdf"))
        hessian = np.loadtxt(fname=get_data("ethane_hessian.txt"))
        mol.hessian = hessian
        mod_sem = ModSeminario(molecule=mol)
        mod_sem.modified_seminario_method()
        # now check the values
        assert mol.HarmonicBondForce[(0, 1)][0] == pytest.approx(0.15344171531887932)
        assert mol.HarmonicBondForce[(0, 1)][1] == pytest.approx(192462.76956156612)
        assert mol.HarmonicBondForce[(0, 2)][0] == pytest.approx(0.10954907576059233)
        assert mol.HarmonicBondForce[(0, 2)][1] == pytest.approx(295645.6124892813)

        assert mol.HarmonicAngleForce[(1, 0, 2)][0] == pytest.approx(1.9423960113296368)
        assert mol.HarmonicAngleForce[(1, 0, 2)][1] == pytest.approx(374.76469990519263)
        assert mol.HarmonicAngleForce[(1, 0, 3)][0] == pytest.approx(1.9422108316309619)
        assert mol.HarmonicAngleForce[(1, 0, 3)][1] == pytest.approx(401.56353614024914)
        assert mol.HarmonicAngleForce[(1, 0, 4)][0] == pytest.approx(1.9416241741805782)
        assert mol.HarmonicAngleForce[(1, 0, 4)][1] == pytest.approx(371.0717571027322)
        assert mol.HarmonicAngleForce[(2, 0, 3)][0] == pytest.approx(1.8787818480998344)
        assert mol.HarmonicAngleForce[(2, 0, 3)][1] == pytest.approx(314.5677633711689)
        assert mol.HarmonicAngleForce[(0, 1, 6)][0] == pytest.approx(1.9423960113296368)
        assert mol.HarmonicAngleForce[(0, 1, 6)][1] == pytest.approx(399.59297576081184)


def test_mod_sem_symmetry(tmpdir):
    """
    A simple regression test for modified seminario method with symmetry.
    """
    with tmpdir.as_cwd():
        # the sdf file is at the qm geometry
        mol = Ligand.from_file(file_name=get_data("ethane.sdf"))
        hessian = np.loadtxt(fname=get_data("ethane_hessian.txt"))
        mol.hessian = hessian
        mod_sem = ModSeminario(molecule=mol)
        mod_sem.modified_seminario_method()
        mod_sem.symmetrise_bonded_parameters()
        # make sure symmetry groups are the same
        for bonds in mol.bond_types.values():
            values = set()
            for bond in bonds:
                values.add(tuple(mol.HarmonicBondForce[bond]))
            assert len(values) == 1
        for angles in mol.angle_types.values():
            values = set()
            for angle in angles:
                values.add(tuple(mol.HarmonicAngleForce[angle]))
                assert len(values) == 1
