"""
All tests are ran on ethane optimised with psi4 at b3lyp-d3bj/dzvp.
"""

import numpy as np
import pytest

from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.molecules import Ligand
from QUBEKit.utils import constants
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
        mod_sem = ModSeminario(symmetrise_parameters=False)
        mod_sem.run(molecule=mol)
        # now check the values
        assert mol.BondForce.get_parameter((0, 1)).length == pytest.approx(
            0.15344171531887932
        )
        assert mol.BondForce.get_parameter((0, 1)).k == pytest.approx(
            192462.76956156612
        )
        assert mol.BondForce.get_parameter((0, 2)).length == pytest.approx(
            0.10954907576059233
        )
        assert mol.BondForce.get_parameter((0, 2)).k == pytest.approx(295645.6124892813)

        assert mol.AngleForce.get_parameter((1, 0, 2)).angle == pytest.approx(
            1.9423960113296368
        )
        assert mol.AngleForce.get_parameter((1, 0, 2)).k == pytest.approx(
            374.76469990519263
        )
        assert mol.AngleForce.get_parameter((1, 0, 3)).k == pytest.approx(
            401.56353614024914
        )
        assert mol.AngleForce.get_parameter((1, 0, 4)).angle == pytest.approx(
            1.9416241741805782
        )
        assert mol.AngleForce.get_parameter((1, 0, 3)).angle == pytest.approx(
            1.9422108316309619
        )
        assert mol.AngleForce.get_parameter((1, 0, 4)).k == pytest.approx(
            371.0717571027322
        )
        assert mol.AngleForce.get_parameter((2, 0, 3)).angle == pytest.approx(
            1.8787818480998344
        )
        assert mol.AngleForce.get_parameter((2, 0, 3)).k == pytest.approx(
            314.5677633711689
        )
        assert mol.AngleForce.get_parameter((0, 1, 6)).angle == pytest.approx(
            1.9423960113296368
        )
        assert mol.AngleForce.get_parameter((0, 1, 6)).k == pytest.approx(
            399.59297576081184
        )


def test_mod_sem_symmetry(tmpdir):
    """
    A simple regression test for modified seminario method with symmetry.
    """
    with tmpdir.as_cwd():
        # the sdf file is at the qm geometry
        mol = Ligand.from_file(file_name=get_data("ethane.sdf"))
        hessian = np.loadtxt(fname=get_data("ethane_hessian.txt"))
        mol.hessian = hessian
        mod_sem = ModSeminario(symmetrise_parameters=True)
        mod_sem.run(molecule=mol)
        # make sure symmetry groups are the same
        for bonds in mol.bond_types.values():
            values = set()
            for bond in bonds:
                parameter = mol.BondForce.get_parameter(atoms=bond)
                values.add(tuple([parameter.length, parameter.k]))
            assert len(values) == 1
        for angles in mol.angle_types.values():
            values = set()
            for angle in angles:
                parameter = mol.AngleForce.get_parameter(atoms=angle)
                values.add(tuple([parameter.angle, parameter.k]))
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
        mod_sem = ModSeminario(symmetrise_parameters=False)
        mod_sem.run(molecule=mol)

        # check the C-CL bond
        parameter = mol.BondForce.get_parameter(atoms=(0, 1))
        assert round(parameter.length, ndigits=4) == 0.1805
        # get in kcal/mol like the reference values
        assert round(parameter.k * constants.KJ_TO_KCAL / 200, 2) == 155.640
        # check a C-H bond
        parameter = mol.BondForce.get_parameter(atoms=(0, 2))
        assert round(parameter.length, 3) == 0.109
        assert round(parameter.k * constants.KJ_TO_KCAL / 200, 2) == 379.69

        # check a CL-C-H angle
        parameter = mol.AngleForce.get_parameter(atoms=(1, 0, 2))
        assert round(parameter.angle * constants.RAD_TO_DEG, 2) == 108.50
        assert round(parameter.k * constants.KJ_TO_KCAL / 2, 2) == 38.16

        # check a H-C-H angle
        parameter = mol.AngleForce.get_parameter(atoms=(2, 0, 3))
        assert round(parameter.angle * constants.RAD_TO_DEG, 2) == 110.31
        assert round(parameter.k * constants.KJ_TO_KCAL / 2, 2) == 37.53
