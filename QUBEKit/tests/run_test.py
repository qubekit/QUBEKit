"""
Test each stage in run with multiple input options to make sure configs are handled.
"""
import os
import shutil

import pytest

from QUBEKit.ligand import Ligand
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
        mol.name = "pyridine"
        mol.parameter_engine = parameter_engine
        if parameter_engine == "xml":
            shutil.copy(get_data("pyridine.xml"), "pyridine.xml")
        param_mol = Execute.parametrise(molecule=mol, verbose=False)
        # make sure parameters have been found
        for i in range(param_mol.n_atoms):
            assert param_mol.NonbondedForce[i] != [0, 0, 0]


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


def test_parametrise_none(tmpdir):
    """
    If no engine is passed make sure we init the parameter holders but store nothing.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("acetone.pdb"))
        mol.parameter_engine = "none"
        param_mol = Execute.parametrise(molecule=mol, verbose=False)
        for i in range(param_mol.n_atoms):
            assert param_mol.NonbondedForce[i] == [0, 0, 0]
