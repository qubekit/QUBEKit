#!/usr/bin/env python3

import numpy as np
import pytest

from QUBEKit.ligand import Ligand
from QUBEKit.utils.constants import BOHR_TO_ANGS
from QUBEKit.utils.datastructures import CustomNamespace
from QUBEKit.utils.file_handling import get_data
from QUBEKit.virtual_sites import VirtualSites


@pytest.fixture(scope="module")
def mol():
    """
    Initialise the Ligand molecule object with data for Chloromethane
    """
    molecule = Ligand.from_file(file_name=get_data("chloromethane.pdb"))
    molecule.ddec_data = {
        0: CustomNamespace(
            a_i=72461.2438863321,
            atomic_symbol="C",
            b_i=36.09781017184126,
            charge=-0.220088,
            r_aim=1.9933297947778903,
            volume=30.276517,
        ),
        1: CustomNamespace(
            a_i=153692.84134145387,
            atomic_symbol="Cl",
            b_i=101.44341268889193,
            charge=1.815899,
            r_aim=1.9020122149415648,
            volume=67.413573,
        ),
        2: CustomNamespace(
            a_i=149.1117208173859,
            atomic_symbol="H",
            b_i=1.247688109065071,
            charge=0.13473,
            r_aim=1.2455924332095252,
            volume=3.329737,
        ),
        3: CustomNamespace(
            a_i=149.1117208173859,
            atomic_symbol="H",
            b_i=1.247688109065071,
            charge=0.13473,
            r_aim=1.2455924332095252,
            volume=3.329737,
        ),
        4: CustomNamespace(
            a_i=149.1117208173859,
            atomic_symbol="H",
            b_i=1.247688109065071,
            charge=0.134729,
            r_aim=1.2455924332095252,
            volume=3.329737,
        ),
    }
    molecule.dipole_moment_data = {
        0: CustomNamespace(x_dipole=0.109154, y_dipole=0.006347, z_dipole=-0.000885),
        1: CustomNamespace(x_dipole=-0.139599, y_dipole=-0.006372, z_dipole=0.000994),
        2: CustomNamespace(x_dipole=-0.005778, y_dipole=-0.018142, z_dipole=-0.029462),
        3: CustomNamespace(x_dipole=-0.00516, y_dipole=-0.016898, z_dipole=0.030335),
        4: CustomNamespace(x_dipole=-0.00839, y_dipole=0.035106, z_dipole=-0.000628),
    }

    molecule.quadrupole_moment_data = {
        0: CustomNamespace(
            q_3z2_r2=-0.150042,
            q_x2_y2=0.148149,
            q_xy=0.007494,
            q_xz=-0.001301,
            q_yz=-0.000128,
        ),
        1: CustomNamespace(
            q_3z2_r2=-1.074695,
            q_x2_y2=1.070914,
            q_xy=0.052325,
            q_xz=-0.006765,
            q_yz=-0.000286,
        ),
        2: CustomNamespace(
            q_3z2_r2=0.013971,
            q_x2_y2=0.011282,
            q_xy=0.001128,
            q_xz=0.000274,
            q_yz=0.011593,
        ),
        3: CustomNamespace(
            q_3z2_r2=0.01544,
            q_x2_y2=0.011683,
            q_xy=0.001125,
            q_xz=-0.000412,
            q_yz=-0.01131,
        ),
        4: CustomNamespace(
            q_3z2_r2=-0.043386,
            q_x2_y2=-0.007519,
            q_xy=-0.001058,
            q_xz=-5.4e-05,
            q_yz=-0.000249,
        ),
    }

    molecule.cloud_pen_data = {
        0: CustomNamespace(a=2.102843, atomic_symbol="C", b=2.40575),
        1: CustomNamespace(a=7.939831, atomic_symbol="Cl", b=3.395079),
        2: CustomNamespace(a=0.1242, atomic_symbol="H", b=2.533532),
        3: CustomNamespace(a=0.123448, atomic_symbol="H", b=2.533309),
        4: CustomNamespace(a=0.120282, atomic_symbol="H", b=2.533191),
    }
    print(f"mol coords: {molecule.coordinates}")
    return molecule


@pytest.fixture(scope="module")
def vs(mol):
    """
    Initialise the VirtualSites class to be used for the following tests
    """
    virtual = VirtualSites(mol, debug=True)
    print(f"coords: {virtual.coords}")
    return VirtualSites(mol, debug=True)


@pytest.mark.parametrize(
    "input_array, result",
    [
        pytest.param(np.array([0, 0, 0]), np.array([0, 0, 0]), id="origin"),
        pytest.param(
            np.array([1, 1, 1]),
            np.array([0.45464871, 0.70807342, 0.54030231]),
            id="unit conversion",
        ),
    ],
)
def test_spherical_to_cartesian(input_array, result, vs):
    for i in range(3):
        assert vs.spherical_to_cartesian(input_array)[i] == pytest.approx(result[i])


@pytest.mark.parametrize(
    "array1, array2, result",
    [
        pytest.param(np.array([1, 2, 3]), np.array([1, 2, 3]), 0, id="no dist"),
        pytest.param(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            pytest.approx(np.sqrt(3)),
            id="unit distance",
        ),
    ],
)
def test_xyz_distance(array1, array2, result, vs):
    assert vs.xyz_distance(array1, array2) == result


@pytest.mark.parametrize(
    "charge, dist, result",
    [
        pytest.param(0, 1, 0, id="no charge"),
        pytest.param(1, 1, pytest.approx(2.307077552e-28), id="charge 1, dist 1"),
    ],
)
def test_monopole_esp_one_charge(charge, dist, result, vs):
    assert vs.monopole_esp_one_charge(charge, dist) == result


def test_monopole_esp_one_charge_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs.monopole_esp_one_charge(1, 0)


@pytest.mark.parametrize(
    "charge1, charge2, dist1, dist2, result",
    [
        pytest.param(0, 0, 1, 1, 0, id="no charges"),
        pytest.param(
            1, 1, 1, 1, pytest.approx(4.614155105e-28), id="charges 1, dists 1"
        ),
    ],
)
def test_monopole_esp_two_charges(charge1, charge2, dist1, dist2, result, vs):
    assert vs.monopole_esp_two_charges(charge1, charge2, dist1, dist2) == result


def test_monopole_esp_two_charges_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs.monopole_esp_two_charges(1, 1, 0, 0)


@pytest.mark.parametrize(
    "charge1, charge2, charge3, dist1, dist2, dist3, result",
    [
        pytest.param(0, 0, 0, 1, 1, 1, 0, id="no charges"),
        pytest.param(
            1, 1, 1, 1, 1, 1, pytest.approx(6.921232657e-28), id="charges 1, dists 1"
        ),
    ],
)
def test_monopole_esp_three_charges(
    charge1, charge2, charge3, dist1, dist2, dist3, result, vs
):
    assert (
        vs.monopole_esp_three_charges(charge1, charge2, charge3, dist1, dist2, dist3)
        == result
    )


def test_monopole_esp_three_charges_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs.monopole_esp_three_charges(1, 1, 1, 0, 0, 0)


def test_dipole_esp(mol, vs):
    dip_data = mol.dipole_moment_data[1]
    dipole_moment = np.array([*dip_data.values()]) * BOHR_TO_ANGS

    assert vs.dipole_esp(np.array([1, 1, 1]), dipole_moment, 1) == pytest.approx(
        -1.76995515193e-29
    )


def test_quadrupole_esp(mol, vs):
    quad_data = mol.quadrupole_moment_data[1]
    m_tensor = vs.quadrupole_moment_tensor(*quad_data.values())
    assert vs.quadrupole_esp(np.array([1, 1, 1]), m_tensor, 1) == pytest.approx(
        9.40851165275e-30
    )


def test_cloud_penetration(mol, vs):
    cloud_pen_data = mol.cloud_pen_data[1]
    a, b = cloud_pen_data.a, cloud_pen_data.b
    b /= BOHR_TO_ANGS
    assert vs.cloud_penetration(a, b, 1) == pytest.approx(2.86224473231e-27)


def test_generate_sample_points_relative(vs):
    points = vs.generate_sample_points_relative(vdw_radius=1)
    for point in points:
        # All points should be 1.4-2.0x the vdw radius (in this instance, 1 Ang)
        assert 1.39 <= vs.xyz_distance(point, np.array([0, 0, 0])) <= 2.01


def test_get_vector_from_coords(vs):
    vector = vs.get_vector_from_coords(atom_index=1, n_sites=1, alt=False)
    # Chlorine scale factor == 1.5
    assert np.linalg.norm(vector) == pytest.approx(1.5)


# def test_fit(vs):
#     vs.sample_points = vs.generate_sample_points_atom(1)
#     vs.no_site_esps = vs.generate_esp_atom(1)
#     vs.fit(1)
#
#     assert len(vs.v_sites_coords) != 0
#     for err in vs.site_errors.values():
#         # placeholder errors are
#         assert err % 5 != 0
