import pytest
import qcengine

from qubekit.engines import call_qcengine
from qubekit.utils.datastructures import LocalResource, QCOptions, TDSettings


@pytest.mark.parametrize(
    "qc_options",
    [
        pytest.param(
            QCOptions(program="rdkit", basis=None, method="mmff94"), id="rdkit mmff"
        ),
        pytest.param(
            QCOptions(program="openmm", basis="smirnoff", method="openff-1.0.0.offxml"),
            id="parsley",
        ),
        pytest.param(
            QCOptions(program="openmm", basis="antechamber", method="gaff-2.11"),
            id="gaff-2.11",
        ),
        pytest.param(
            QCOptions(program="psi4", basis="3-21g", method="hf"), id="psi4 hf"
        ),
        pytest.param(
            QCOptions(program="gaussian", basis="3-21g", method="hf"), id="gaussian hf"
        ),
    ],
)
def test_single_point_energy(qc_options: QCOptions, tmpdir, water):
    """
    Make sure our qcengine wrapper works correctly.
    """
    if qc_options.program.lower() not in qcengine.list_available_programs():
        pytest.skip(f"{qc_options.program} missing skipping test.")

    with tmpdir.as_cwd():
        result = call_qcengine(
            molecule=water,
            driver="energy",
            qc_spec=qc_options,
            local_options=LocalResource(cores=1, memory=1),
        )
        assert result.driver == "energy"
        assert result.model.basis == qc_options.basis
        assert result.model.method == qc_options.method
        assert result.provenance.creator.lower() == qc_options.program
        # make sure the grid was set to ultrafine for psi4
        if qc_options.program == "psi4":
            assert result.keywords["dft_spherical_points"] == 590
            assert result.keywords["dft_radial_points"] == 99


def test_psi4_tddft(water, tmpdir):
    """
    Make sure td settings are respected and used in the calculation.
    """
    with tmpdir.as_cwd():
        qc_spec = QCOptions(
            method="td-hf",
            basis="3-21g",
            td_settings=TDSettings(n_states=3, use_tda=True),
        )
        result = call_qcengine(
            molecule=water,
            driver="energy",
            qc_spec=qc_spec,
            local_options=LocalResource(cores=1, memory=1),
        )
        assert (
            "TD-HF ROOT 0 -> ROOT 1 MAGNETIC TRANSITION DIPOLE MOMENT - A SYMMETRY"
            in result.extras["qcvars"]
        )


@pytest.mark.parametrize(
    "functional, result",
    [
        pytest.param("hf", "td-hf", id="hf"),
        pytest.param("td-cam-b3lyp", "td-cam-b3lyp", id="cam-b3lyp"),
    ],
)
def test_td_method_conversion_psi4(functional, result):
    """
    Make sure functionals are converted correctly when we have TD settings.
    """
    spec = QCOptions(
        program="psi4",
        method=functional,
        basis="6-31G",
        td_settings=TDSettings(use_tda=True, n_states=3),
    )
    qc_model = spec.qc_model
    assert qc_model.method == result


def test_td_method_conversion_gaussian():
    """
    Make sure we do not convert the function even with td settings when using gaussian.
    """
    spec = QCOptions(
        program="gaussian",
        method="cam-b3lyp",
        basis="6-31G",
        td_settings=TDSettings(use_tda=True, n_states=3),
    )
    qc_model = spec.qc_model
    assert qc_model.method == "cam-b3lyp"
