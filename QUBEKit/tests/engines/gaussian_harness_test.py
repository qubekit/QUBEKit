"""
Tests for the qcengine gaussian harness
"""

import numpy as np
import pytest
import qcelemental as qcel
import qcengine as qcng

from QUBEKit.engines.gaussian_harness import GaussianHarness
from QUBEKit.molecules import Ligand
from QUBEKit.utils.file_handling import get_data


def test_gaussian_version():
    """
    Try and check the gaussian version this will raise an error if not installed.
    """
    g = GaussianHarness()
    if g.found():
        assert g.get_version() in ["g09", "g16"]
    else:
        with pytest.raises(ModuleNotFoundError):
            _ = g.get_version()


@pytest.mark.parametrize(
    "function, result",
    [
        pytest.param("pbe", "PBEPBE", id="pbe"),
        pytest.param("wb97x-d", "wB97XD", id="wb97xd"),
        pytest.param("b3lyp", "b3lyp", id="b3lyp"),
    ],
)
def test_functional_converter(function, result):
    """
    Make sure that problem functions are converted.
    """
    theory = GaussianHarness.functional_converter(method=function)
    assert theory == result


@pytest.mark.parametrize(
    "functional, result",
    [
        pytest.param("b3lyp-d3bj", "EmpiricalDispersion=GD3BJ b3lyp", id="b3lyp-d3bj"),
        pytest.param("pbe-d3", "EmpiricalDispersion=GD3 PBEPBE", id="pbe-d3"),
    ],
)
def test_dispersion_converter(functional, result):
    """
    Make sure that dispersion corrections are converted.
    """
    theory = GaussianHarness.functional_converter(method=functional)
    assert result == theory


@pytest.mark.parametrize(
    "driver, result",
    [
        pytest.param("energy", "SP", id="Energy"),
        pytest.param("gradient", "Force=NoStep", id="gradient"),
        pytest.param("hessian", "FREQ", id="hessian"),
    ],
)
def test_driver_conversion(driver, result):
    """
    Make sure each of the drivers are converted correctly.
    """
    g_driver = GaussianHarness.driver_conversion(driver=driver)
    assert g_driver == result


@pytest.mark.parametrize(
    "driver, result",
    [
        pytest.param("energy", "SP", id="energy"),
        pytest.param("gradient", "Force=NoStep", id="gradient"),
        pytest.param("hessian", "FREQ", id="hessian"),
    ],
)
def test_build_input(tmpdir, driver, result):
    """
    Build a gaussian input file for the given input model.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("acetone.pdb"))
        # build the atomic model
        qc_spec = qcel.models.common_models.Model(method="pbe", basis="6-31G")
        # build a job for s specific driver
        qc_task = qcel.models.AtomicInput(
            molecule=mol.to_qcschema(), driver=driver, model=qc_spec
        )
        g = GaussianHarness()
        local_options = qcng.config.TaskConfig(
            ncores=10, memory=10, nnodes=1, retries=1
        )
        input_data = g.build_input(input_model=qc_task, config=local_options)
        file_input = input_data["infiles"]["gaussian.com"]
        # now make sure the driver conversion is in the file input
        assert result in file_input
        # make sure we can write the file with no issues
        with open("gaussian.com", "w") as com:
            com.write(file_input)


def test_get_version_from_log():
    """
    Try and extract the version info from a mock log file.
    """
    # mock output file with version stamp
    with open(get_data("gaussian.log")) as log:
        logfile = log.read()

    version = GaussianHarness.parse_version(logfile=logfile)
    assert version == "Gaussian 09:  ES64L-G09RevD.01 24-Apr-2013"


def test_parse_gradient():
    """
    Make sure we can parse the gradient from a fchk file.
    """
    with open(get_data("gaussian.fchk")) as fchk:
        fchkfile = fchk.read()

    gradient = GaussianHarness.parse_gradient(fchfile=fchkfile)
    assert len(gradient) == 30
    # check the first and last values
    assert gradient[0] == -1.01049212e-02
    assert gradient[-1] == -7.31912443e-03


def test_parse_hessian():
    """
    Make sure we can parse the hessian from a g09 mock fchk file.
    """
    with open(get_data("gaussian.fchk")) as fchk:
        fchkfile = fchk.read()

    hessian = GaussianHarness.parse_hessian(fchkfile=fchkfile)
    assert len(hessian) == 900


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param("energy", id="energy"),
        pytest.param("gradient", id="gradient"),
        pytest.param("hessian", id="hessian"),
    ],
)
def test_parse_output(driver):
    """
    Test reading gaussian outfiles and extracting the correct information based on the
    driver type.
    """
    outfiles = {}
    with open(get_data("gaussian.log")) as log:
        outfiles["gaussian.log"] = log.read()
    with open(get_data("gaussian.fchk")) as fchk:
        outfiles["lig.fchk"] = fchk.read()

    # build the input
    mol = Ligand.from_file(file_name=get_data("acetone.pdb"))
    # build the atomic model
    qc_spec = qcel.models.common_models.Model(method="pbe", basis="6-31G")
    # build a job for a specific driver
    qc_task = qcel.models.AtomicInput(
        molecule=mol.to_qcschema(), driver=driver, model=qc_spec
    )
    g = GaussianHarness()
    result = g.parse_output(outfiles=outfiles, input_model=qc_task)
    if driver == "energy":
        assert result.return_result == -1.931393770857046e02
    elif driver == "gradient":
        assert result.return_result.shape == (10, 3)
    elif driver == "hessian":
        assert result.return_result.shape == (30, 30)


def test_fail_termination():
    """
    If we think the gaussian job did not finish then we need to make sure an error is raised.
    """
    random_string = "test string\n test string"
    with pytest.raises(qcng.exceptions.UnknownError):
        GaussianHarness.check_convergence(logfile=random_string)


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param("energy", id="energy"),
        pytest.param("gradient", id="gradient"),
        pytest.param("hessian", id="hessian"),
    ],
)
def test_full_run(driver, tmpdir):
    """
    For the given driver try a full execution if the user has gaussian installed.
    """
    if not GaussianHarness.found():
        pytest.skip("Gaussian 09/16 not available test skipped.")

    with tmpdir.as_cwd():
        # build the input
        mol = Ligand.from_file(file_name=get_data("acetone.pdb"))
        # build the atomic model
        qc_spec = qcel.models.common_models.Model(
            method="wB97XD", basis="6-311++G(d,p)"
        )
        # build a job for a specific driver
        qc_task = qcel.models.AtomicInput(
            molecule=mol.to_qcschema(), driver=driver, model=qc_spec
        )
        g = GaussianHarness()
        # run locally with 2 cores and 2 GB memory
        result = g.compute(
            input_data=qc_task,
            config=qcng.config.TaskConfig(
                **{"memory": 2, "ncores": 2, "nnodes": 1, "retries": 1}
            ),
        )

        outfiles = {}
        with open(get_data("gaussian.log")) as log:
            outfiles["gaussian.log"] = log.read()
        with open(get_data("gaussian.fchk")) as fchk:
            outfiles["lig.fchk"] = fchk.read()
        ref_result = g.parse_output(outfiles=outfiles, input_model=qc_task)

        assert np.allclose(ref_result.return_result, result.return_result)
