import pytest

from qubekit.molecules import Ligand
from qubekit.parametrisation import XML, AnteChamber, OpenFF
from qubekit.utils.file_handling import get_data


@pytest.fixture
def acetone():
    """
    Make a ligand class from the acetone pdb.
    """
    return Ligand.from_file(file_name=get_data("acetone.sdf"))


@pytest.fixture
def water():
    """Make a qube water molecule."""
    return Ligand.from_file(file_name=get_data("water.pdb"))


@pytest.fixture
def antechamber():
    return AnteChamber(force_field="gaff2")


@pytest.fixture
def openff():
    return OpenFF(force_field="openff_unconstrained-1.3.0.offxml")


@pytest.fixture
def xml():
    return XML()


@pytest.fixture
def coumarin():
    return Ligand.parse_file(get_data("coumarin_hess_wbo.json"))
