import os

import pytest
import qcengine

from qubekit.engines import GeometryOptimiser
from qubekit.molecules import Ligand
from qubekit.utils.exceptions import SpecificationError
from qubekit.utils.file_handling import get_data


def test_rdkit_available():
    """
    Make sure the geometry optimiser allows rdkit as this comes with QUBEKit.
    """
    g_opt = GeometryOptimiser()
    g_opt.program = "rdkit"


def test_program_not_installed():
    """
    Make sure an error is raised when we try and use a program that is not available.
    """
    g_opt = GeometryOptimiser()
    with pytest.raises(SpecificationError):
        g_opt.program = "test"


def test_local_options():
    """
    Make sure the task config is correctly made and keywords are converted.
    """
    g_opt = GeometryOptimiser(cores=10, memory=2)
    local_options = g_opt.local_options
    assert g_opt.cores == local_options["ncores"]
    assert g_opt.memory == local_options["memory"]


@pytest.mark.parametrize(
    "optimiser",
    [
        pytest.param("geometric", id="geometric"),
        pytest.param("optking", id="optking"),
    ],
)
def test_optimiser_keywords(optimiser):
    """
    For the given optimiser make sure the keywords are updated correctly.
    """
    if "psi4" not in qcengine.list_available_programs():
        pytest.skip("Psi4 missing skipping.")
    g = GeometryOptimiser(
        optimiser=optimiser, maxiter=1, convergence="GAU", program="psi4"
    )
    keywords = g.build_optimiser_keywords()
    assert 1 in keywords.values()
    assert "GAU" in keywords.values()


def test_missing_optimiser():
    """
    Make sure an error is raised when we try and set a missing optimiser.
    """
    with pytest.raises(SpecificationError):
        _ = GeometryOptimiser(optimiser="bad_optimiser")


@pytest.mark.parametrize(
    "program, basis, method",
    [
        pytest.param("rdkit", None, "UFF", id="rdkit uff"),
        pytest.param("psi4", "6-311G", "b3lyp", id="psi4 b3lyp"),
        pytest.param("openmm", "smirnoff", "openff_unconstrained-1.3.0", id="openff"),
        pytest.param("openmm", "antechamber", "gaff-2.11", id="gaff"),
    ],
)
def test_spec_validation_pass(program, basis, method):
    """
    Make sure we can correctly validate a specification.
    """
    if program not in qcengine.list_available_programs():
        pytest.skip(f"{program} missing skipping test")
    _ = GeometryOptimiser(program=program, basis=basis, method=method)


@pytest.mark.parametrize(
    "program, basis, method",
    [
        pytest.param("rdkit", "uff", "uff", id="rdkit wrong basis"),
        pytest.param("openmm", "gaff", "gaff-2.11", id="openmm wrong basis"),
        pytest.param("openmm", "smirnoff", "openff-sage", id="openff wrong method."),
    ],
)
def test_spec_validation_fail(program, basis, method):
    """
    Make sure than an invalid specification raises an error.
    """
    if program not in qcengine.list_available_programs():
        pytest.skip(f"{program} missing skipping test")
    with pytest.raises(SpecificationError):
        _ = GeometryOptimiser(program=program, basis=basis, method=method)


@pytest.mark.parametrize(
    "program, basis, method",
    [
        pytest.param("rdkit", None, "uff", id="rdkit uff"),
        pytest.param("openmm", "antechamber", "gaff-2.11", id="gaff-2.11"),
        pytest.param(
            "openmm", "smirnoff", "openff_unconstrained-1.3.0", id="openff 1.3.0"
        ),
        pytest.param("psi4", "3-21g", "hf", id="psi4 hf"),
        pytest.param("gaussian", "3-21g", "hf", id="gaussian hf"),
    ],
)
def test_optimise(program, basis, method, tmpdir):
    """
    Test running different optimisers with different programs.
    """
    if program not in qcengine.list_available_programs():
        pytest.skip(f"{program} missing skipping test.")

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("water.pdb"))
        g = GeometryOptimiser(
            program=program,
            basis=basis,
            method=method,
            optimiser="geometric",
            convergence="GAU",
        )
        result_mol, _ = g.optimise(molecule=mol, return_result=False)
        assert result_mol.coordinates.tolist() != mol.coordinates.tolist()


def test_optimise_fail_output(tmpdir):
    """
    Make sure the optimised geometries and result is still wrote out if we fail the molecule and an error is rasied.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        g = GeometryOptimiser(program="rdkit", method="uff", basis=None, maxiter=2)
        with pytest.raises(RuntimeError):
            g.optimise(molecule=mol, allow_fail=False)
        files = os.listdir()
        assert "opt.xyz" in files
        assert "opt_trajectory.xyz" in files
        assert "result.json" in files


def test_optmiser_fail_no_output(tmpdir):
    """
    Make sure we raise an error correctly when there is no output from a failed optimisation.
    """
    if "psi4" not in qcengine.list_available_programs():
        pytest.skip("Psi4 missing skipping test.")
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        g = GeometryOptimiser(
            program="psi4", method="wb97x-dbj", basis="dzvp", maxiter=10
        )
        with pytest.raises(RuntimeError):
            g.optimise(molecule=mol, allow_fail=False)


def test_optking_fail():
    """
    Optking currently only works with psi4 make sure we raise an error if we use
    a different program.
    """
    with pytest.raises(SpecificationError):
        GeometryOptimiser(optimiser="optking", program="rdkit")
