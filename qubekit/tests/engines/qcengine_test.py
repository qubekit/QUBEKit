import pytest
import qcengine

from qubekit.engines import QCEngine
from qubekit.molecules import Ligand
from qubekit.utils.file_handling import get_data


@pytest.mark.parametrize(
    "program, basis, method",
    [
        pytest.param("rdkit", None, "mmff94", id="rdkit mmff"),
        pytest.param("xtb", None, "gfn2xtb", id="gfn2xtb"),
        pytest.param("openmm", "smirnoff", "openff-1.0.0.offxml", id="parsley"),
        pytest.param("openmm", "antechamber", "gaff-2.11", id="gaff-2.11"),
        pytest.param("torchani", None, "ani2x", id="ani2x"),
        pytest.param("psi4", "3-21g", "hf", id="psi4 hf"),
        pytest.param("gaussian", "3-21g", "hf", id="gaussian hf"),
    ],
)
def test_single_point_energy(program, basis, method, tmpdir):
    """
    Make sure our qcengine wrapper works correctly.
    """
    if program not in qcengine.list_available_programs():
        pytest.skip(f"{program} missing skipping test.")

    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("water.pdb"))
        engine = QCEngine(
            program=program,
            basis=basis,
            method=method,
            memory=1,
            cores=1,
            driver="energy",
        )
        result = engine.call_qcengine(molecule=mol)
        assert result.driver == "energy"
        assert result.model.basis == basis
        assert result.model.method == method
        assert result.provenance.creator.lower() == program
