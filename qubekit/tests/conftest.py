import pytest

from qubekit.molecules import Ligand
from qubekit.parametrisation import XML, AnteChamber, OpenFF
from qubekit.utils.file_handling import get_data
from qubekit.workflow.workflow import QCOptions, WorkFlow


@pytest.fixture()
def acetone():
    """
    Make a ligand class from the acetone pdb.
    """
    return Ligand.from_file(file_name=get_data("acetone.sdf"))


@pytest.fixture()
def water():
    """Make a qube water molecule."""
    return Ligand.from_file(file_name=get_data("water.pdb"))


@pytest.fixture()
def antechamber():
    return AnteChamber(force_field="gaff2")


@pytest.fixture()
def openff():
    return OpenFF(force_field="openff_unconstrained-2.0.0.offxml")


@pytest.fixture()
def xml():
    return XML()


@pytest.fixture()
def coumarin():
    return Ligand.parse_file(get_data("coumarin_hess_wbo.json"))


@pytest.fixture()
def mol_47():
    return Ligand.from_smiles("CC(C)(O)CCC(C)(C)O", "mol_47")


@pytest.fixture()
def rdkit_workflow():
    rdkit_spec = QCOptions(program="rdkit", method="uff", basis=None)
    workflow = WorkFlow(qc_options=rdkit_spec)
    return workflow


@pytest.fixture()
def rfree_data():
    return {
        "H": {"v_free": 7.6, "b_free": 6.5, "r_free": 1.738},
        "X": {"v_free": 7.6, "b_free": 6.5, "r_free": 1.083},
        "C": {"v_free": 34.4, "b_free": 46.6, "r_free": 2.008},
        "N": {"v_free": 25.9, "b_free": 24.2, "r_free": 1.765},
        "O": {"v_free": 22.1, "b_free": 15.6, "r_free": 1.499},
        "F": {"v_free": 18.2, "b_free": 9.5, "r_free": 1.58},
        "Cl": {"v_free": 65.1, "b_free": 94.6, "r_free": 1.88},
        "Br": {"v_free": 95.7, "b_free": 162.0, "r_free": 1.96},
        "S": {"v_free": 75.2, "b_free": 134.0, "r_free": 2.0},
        "alpha": 1,
        "beta": 0.5,
    }


@pytest.fixture()
def methanol():
    return Ligand.parse_file(get_data("methanol.json"))
