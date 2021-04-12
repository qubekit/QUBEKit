#!/usr/bin/env python3

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from qubekit.lennard_jones import LennardJones612
from qubekit.molecules import Ligand
from qubekit.parametrisation import OpenFF
from qubekit.utils.constants import BOHR_TO_ANGS
from qubekit.utils.file_handling import ExtractChargeData, get_data
from qubekit.virtual_sites import VirtualSites


@pytest.fixture(scope="module")
def mol():
    """
    Initialise the Ligand molecule object with data for Chloromethane
    """
    # use temp directory to remove parametrisation files
    with TemporaryDirectory() as temp:
        os.chdir(temp)
        molecule = Ligand.from_file(file_name=get_data("chloromethane.pdb"))
        molecule.home = None
        molecule.enable_symmetry = True
        OpenFF().parametrise_molecule(molecule)
        ddec_file_path = get_data("DDEC6_even_tempered_net_atomic_charges.xyz")
        dir_path = os.path.dirname(ddec_file_path)
        ExtractChargeData.read_files(molecule, dir_path, "chargemol")

        return molecule


def test_extract_charge(mol):

    assert mol.atoms[0].aim.charge == -0.220571
    assert mol.atoms[0].dipole.x == 0.109103
    assert mol.atoms[0].aim.volume == 30.289335


def test_apply_symmetrisation(mol):

    assert mol.atoms[2].aim.charge == mol.atoms[3].aim.charge
    assert mol.atoms[2].aim.volume == mol.atoms[3].aim.volume


@pytest.fixture(scope="module")
def vs(mol):
    """
    Initialise the VirtualSites class to be used for the following tests
    """
    virtual_sites = VirtualSites(mol, debug=True)
    return virtual_sites


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
    dip_data = mol.atoms[1].dipole
    dipole_moment = np.array([dip_data.x, dip_data.y, dip_data.z]) * BOHR_TO_ANGS

    assert vs.dipole_esp(np.array([1, 1, 1]), dipole_moment, 1) == pytest.approx(
        -1.76995515193e-29
    )


def test_quadrupole_esp(mol, vs):
    quad_data = mol.atoms[1].quadrupole
    m_tensor = vs.quadrupole_moment_tensor(
        quad_data.q_xy,
        quad_data.q_xz,
        quad_data.q_yz,
        quad_data.q_x2_y2,
        quad_data.q_3z2_r2,
    )
    assert vs.quadrupole_esp(np.array([1, 1, 1]), m_tensor, 1) == pytest.approx(
        9.40851165275e-30
    )


def test_cloud_penetration(mol, vs):
    cloud_pen_data = mol.atoms[1].cloud_pen
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


def test_fit(mol, vs, tmpdir):
    with tmpdir.as_cwd():
        vs.calculate_virtual_sites()

        assert mol.extra_sites is not None

        LennardJones612(mol).calculate_non_bonded_force()
        mol.fix_net_charge()

        assert (
            sum(param.charge for param in mol.NonbondedForce)
            + sum(site.charge for site in mol.extra_sites)
            == 0
        )
