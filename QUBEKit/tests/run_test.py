"""
Test each stage in run with multiple input options to make sure configs are handled.
"""
import os
import shutil

import numpy as np
import pytest
import qcengine

from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.molecules import Ligand
from QUBEKit.run import Execute
from QUBEKit.utils.file_handling import get_data


@pytest.mark.parametrize(
    "parameter_engine",
    [
        pytest.param("antechamber", id="antechamber"),
        pytest.param("openff", id="openff"),
        pytest.param("xml", id="xml"),
    ],
)
def test_parametrise_all(parameter_engine, tmpdir):
    """
    For each parameter engine make sure the molecule is correctly parameterised.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("pyridine.sdf"))
        mol.parameter_engine = parameter_engine
        if parameter_engine == "xml":
            shutil.copy(get_data("pyridine.xml"), "pyridine.xml")
        param_mol = Execute.parametrise(molecule=mol, verbose=False)
        # make sure parameters have been found
        for i in range(param_mol.n_atoms):
            assert param_mol.NonbondedForce.n_parameters == mol.n_atoms


def test_parametrise_missing_file(tmpdir):
    """
    If a missing file is provided make sure an error is raised.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("acetone.pdb"))
        mol.home = os.getcwd()
        mol.parameter_engine = "xml"
        with pytest.raises(FileNotFoundError):
            _ = Execute.parametrise(molecule=mol, verbose=False)


def test_quick_run_to_seminario(tmpdir):
    """
    Do a quick run through each stage in run up to the modified Seminario stage.
    Run on water with a very small basis.
    """
    if "psi4" not in qcengine.list_available_programs():
        pytest.skip("Psi4 missing skipping test")

    with tmpdir.as_cwd():
        water = Ligand.from_file(get_data("water.pdb"))
        water.home = os.getcwd()
        water.parameter_engine = "openff"
        # parmetrise
        water = Execute.parametrise(molecule=water, verbose=False)
        # pre opt
        water.pre_opt_method = "mmff94"
        water_new_coords = Execute.pre_optimise(molecule=water)
        assert not np.allclose(water.coordinates, water_new_coords.coordinates)
        # qm optimise
        water_new_coords.bonds_engine = "psi4"
        water_new_coords.theory = "HF"
        water_new_coords.basis = "STO-3G"
        water_new_coords.threads = 1
        water_qm = Execute.qm_optimise(molecule=water_new_coords)
        assert not np.allclose(water_qm.coordinates, water_new_coords.coordinates)
        # hessian
        water_hess = Execute.hessian(molecule=water_qm)
        # mod sem
        mod_sem = ModSeminario()
        final_water = mod_sem.run(molecule=water_hess)
        # make sure we have symmetry in parameters
        assert final_water.BondForce[(0, 1)].length == final_water.BondForce[(0, 2)].length
        assert final_water.BondForce[(0, 1)].k == final_water.BondForce[(0, 2)].k
        # make sure they are different from the input
        assert final_water.BondForce[(0, 1)].k != pytest.approx(water.BondForce[(0, 1)].k)
