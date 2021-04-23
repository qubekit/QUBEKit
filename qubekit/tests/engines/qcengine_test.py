import pytest
import qcengine

from qubekit.engines import call_qcengine
from qubekit.molecules import Ligand
from qubekit.utils.datastructures import LocalResource, QCOptions
from qubekit.utils.file_handling import get_data


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
def test_single_point_energy(qc_options: QCOptions, tmpdir):
    """
    Make sure our qcengine wrapper works correctly.
    """
    if qc_options.program.lower() not in qcengine.list_available_programs():
        pytest.skip(f"{qc_options.program} missing skipping test.")

    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        result = call_qcengine(
            molecule=mol,
            driver="energy",
            qc_spec=qc_options,
            local_options=LocalResource(cores=1, memory=1),
        )
        assert result.driver == "energy"
        assert result.model.basis == qc_options.basis
        assert result.model.method == qc_options.method
        assert result.provenance.creator.lower() == qc_options.program
