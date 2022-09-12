import os

import pytest
import qcengine

from qubekit.engines import GeometryOptimiser
from qubekit.molecules import Ligand
from qubekit.utils.datastructures import LocalResource, QCOptions
from qubekit.utils.exceptions import SpecificationError
from qubekit.utils.file_handling import get_data


def test_rdkit_available():
    """
    Make sure the geometry optimiser allows rdkit as this comes with QUBEKit.
    """
    qc_ops = QCOptions(program="rdkit", method="uff", basis=None)
    qc_ops.validate_program()


def test_program_not_installed():
    """
    Make sure an error is raised when we try and use a program that is not available.
    """
    g_ops = QCOptions()
    with pytest.raises(SpecificationError):
        g_ops.program = "test"
        g_ops.validate_program()


def test_local_options():
    """
    Make sure the task config is correctly made and keywords are converted.
    """
    l_ops = LocalResource(cores=60, memory=10)
    local_options = l_ops.local_options
    assert l_ops.cores == local_options["ncores"]
    assert l_ops.memory == local_options["memory"]


@pytest.mark.parametrize(
    "optimiser",
    [
        pytest.param("geometric", id="geometric"),
    ],
)
def test_optimiser_keywords(optimiser):
    """
    For the given optimiser make sure the keywords are updated correctly.
    """
    g = GeometryOptimiser(optimiser=optimiser, maxiter=1, convergence="GAU")
    keywords = g._build_optimiser_keywords(program="psi4")
    assert 1 in keywords.values()
    assert "GAU" in keywords.values()


def test_missing_optimiser():
    """
    Make sure an error is raised when we try and set a missing optimiser.
    """
    with pytest.raises(SpecificationError):
        _ = GeometryOptimiser(optimiser="bad_optimiser")


@pytest.mark.parametrize(
    "qc_spec",
    [
        pytest.param(
            QCOptions(program="rdkit", basis=None, method="UFF"), id="rdkit uff"
        ),
        pytest.param(
            QCOptions(program="torchani", basis=None, method="ani2x"), id="ani2x"
        ),
        pytest.param(
            QCOptions(program="psi4", basis="6-311G", method="b3lyp"), id="psi4 b3lyp"
        ),
        pytest.param(
            QCOptions(
                program="openmm", basis="smirnoff", method="openff_unconstrained-1.3.0"
            ),
            id="openff",
        ),
        pytest.param(
            QCOptions(program="openmm", basis="antechamber", method="gaff-2.11"),
            id="gaff",
        ),
    ],
)
def test_spec_validation_pass(qc_spec: QCOptions):
    """
    Make sure we can correctly validate a specification.
    """
    if qc_spec.program.lower() not in qcengine.list_available_programs():
        pytest.skip(f"{qc_spec.program} missing skipping test")
        qc_spec.validate_specification()


@pytest.mark.parametrize(
    "qc_spec",
    [
        pytest.param(
            QCOptions(program="rdkit", basis="uff", method="uff"),
            id="rdkit wrong basis",
        ),
        pytest.param(
            QCOptions(program="openmm", basis="gaff", method="gaff-2.11"),
            id="openmm wrong basis",
        ),
        pytest.param(
            QCOptions(program="xtb", basis="gfn", method="gfn2xtb"),
            id="xtb wrong basis",
        ),
        pytest.param(
            QCOptions(program="torchani", basis=None, method="ani3y"),
            id="torchani wrong method",
        ),
        pytest.param(
            QCOptions(program="openmm", basis="smirnoff", method="openff-sage"),
            id="openff wrong method.",
        ),
    ],
)
def test_spec_validation_fail(qc_spec: QCOptions):
    """
    Make sure than an invalid specification raises an error.
    """
    with pytest.raises(SpecificationError):
        qc_spec.validate_specification()


@pytest.mark.parametrize(
    "qc_spec",
    [
        pytest.param(
            QCOptions(program="rdkit", basis=None, method="uff"), id="rdkit uff"
        ),
        pytest.param(
            QCOptions(program="openmm", basis="antechamber", method="gaff-2.11"),
            id="gaff-2.11",
        ),
        pytest.param(
            QCOptions(
                program="openmm", basis="smirnoff", method="openff_unconstrained-1.3.0"
            ),
            id="openff 1.3.0",
        ),
        pytest.param(
            QCOptions(program="xtb", basis=None, method="gfn2xtb"), id="xtb gfn2xtb"
        ),
        pytest.param(
            QCOptions(program="torchani", basis=None, method="ani1ccx"),
            id="torchani ccx",
        ),
        pytest.param(
            QCOptions(program="psi4", basis="3-21g", method="hf"), id="psi4 hf"
        ),
        pytest.param(
            QCOptions(program="gaussian", basis="3-21g", method="hf"), id="gaussian hf"
        ),
    ],
)
def test_optimise(qc_spec: QCOptions, tmpdir):
    """
    Test running different optimisers with different programs.
    """
    if qc_spec.program.lower() not in qcengine.list_available_programs():
        pytest.skip(f"{qc_spec.program} missing skipping test.")

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("water.pdb"))
        g = GeometryOptimiser(
            optimiser="geometric",
            convergence="GAU",
        )
        result_mol, _ = g.optimise(
            molecule=mol,
            return_result=False,
            qc_spec=qc_spec,
            local_options=LocalResource(cores=1, memory=1),
        )
        assert result_mol.coordinates.tolist() != mol.coordinates.tolist()


def test_optimise_fail_output(tmpdir, water):
    """
    Make sure the optimised geometries and result is still wrote out if we fail the molecule and an error is rasied.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        g = GeometryOptimiser(maxiter=5)
        qc_spec = QCOptions(program="torchani", basis=None, method="ani1ccx")
        with pytest.raises(RuntimeError):
            g.optimise(
                molecule=mol,
                allow_fail=False,
                qc_spec=qc_spec,
                local_options=LocalResource(cores=1, memory=1),
            )
        files = os.listdir()
        assert "opt_water.xyz" in files
        assert "opt_trajectory_water.xyz" in files
        assert "result.json" in files


def test_optmiser_fail_no_output(tmpdir):
    """
    Make sure we raise an error correctly when there is no output from a failed optimisation.
    """
    if "psi4" not in qcengine.list_available_programs():
        pytest.skip("Psi4 missing skipping test.")
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        qc_spec = QCOptions(program="psi4", method="wb97x-dbj", basis="dzvp")
        g = GeometryOptimiser(maxiter=10)
        with pytest.raises(RuntimeError):
            g.optimise(
                molecule=mol,
                allow_fail=False,
                qc_spec=qc_spec,
                local_options=LocalResource(cores=1, memory=1),
            )


def test_optking_fail():
    """
    Optking currently only works with psi4 make sure we raise an error if we use
    a different program.
    """
    with pytest.raises(SpecificationError):
        _ = GeometryOptimiser(optimiser="optking")
