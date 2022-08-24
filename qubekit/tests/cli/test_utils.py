import pytest
from openmm import unit

from qubekit.cli.utils import LocalVirtualSite


def test_local_site_to_dict():
    """Make sure our plugin vsite can be wrote to dict"""

    vsite = LocalVirtualSite(
        name="test",
        orientations=[(0, 1, 2)],
        p1=0 * unit.angstrom,
        p2=0 * unit.angstrom,
        p3=0 * unit.angstrom,
    )
    vsite_dict = vsite.to_dict()
    assert vsite_dict["name"] == "test"
    assert vsite_dict["p1"] == 0 * unit.nanometer


def test_local_site_from_dict_fail():
    """Make sure our plugin site correctly raises an error when we try and make it from the wrong info"""

    with pytest.raises(AssertionError):
        LocalVirtualSite.from_dict({"vsite_type": "BondChargeVirtualSite"})


def test_local_site_from_dict():
    """Make sure our plugin vsite can be correctly loaded from a dict of data."""

    vsite_dict = {
        "name": "LP_from_dict",
        "vsite_type": "LocalVirtualSite",
        "p1": 1 * unit.angstrom,
        "p2": 0 * unit.angstrom,
        "p3": 1 * unit.angstrom,
        "orientations": [(0, 1, 2)],
    }
    lp = LocalVirtualSite.from_dict(vsite_dict=vsite_dict)
    ref_lp = LocalVirtualSite(
        name="LP_from_dict",
        orientations=[(0, 1, 2)],
        p1=0.1 * unit.nanometer,
        p2=0 * unit.nanometer,
        p3=0.1 * unit.nanometer,
    )
    assert ref_lp == lp


def test_local_frame_position():
    """Make sure the local frame position is returned to the correct format"""

    # use angstrom as input to make sure the units are converted to the correct defaults
    lp = LocalVirtualSite(
        name="lp",
        orientations=[(0, 1, 2)],
        p1=1 * unit.angstrom,
        p2=1 * unit.angstrom,
        p3=0 * unit.angstrom,
    )
    assert lp.local_frame_position == unit.Quantity([0.1, 0.1, 0], unit=unit.nanometer)
