#!/usr/bin/env python3

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from qubekit.charges import DDECCharges, ExtractChargeData
from qubekit.molecules import Ligand
from qubekit.nonbonded.protocols import cl_base, get_protocol
from qubekit.nonbonded.virtual_sites import VirtualSites
from qubekit.parametrisation import OpenFF
from qubekit.utils.constants import BOHR_TO_ANGS
from qubekit.utils.file_handling import get_data


@pytest.fixture(scope="module")
def mol():
    """
    Initialise the Ligand molecule object with data for Chloromethane
    """
    # use temp directory to remove parametrisation files
    with TemporaryDirectory() as temp:
        os.chdir(temp)
        molecule = Ligand.from_file(file_name=get_data("chloromethane.pdb"))
        OpenFF().run(molecule)
        ddec_file_path = get_data("DDEC6_even_tempered_net_atomic_charges.xyz")
        dir_path = os.path.dirname(ddec_file_path)
        ExtractChargeData.extract_charge_data_chargemol(molecule, dir_path, 6)
        # apply symmetry to the reference data
        DDECCharges.apply_symmetrisation(molecule=molecule)
        # apply the reference charge to the nonbonded
        for atom in molecule.atoms:
            molecule.NonbondedForce[(atom.atom_index,)].charge = atom.aim.charge

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
    virtual_sites = VirtualSites()
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
        assert vs._spherical_to_cartesian(input_array)[i] == pytest.approx(result[i])


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
    assert vs._xyz_distance(array1, array2) == result


@pytest.mark.parametrize(
    "charge, dist, result",
    [
        pytest.param(0, 1, 0, id="no charge"),
        pytest.param(1, 1, pytest.approx(2.307077552e-28), id="charge 1, dist 1"),
    ],
)
def test_monopole_esp_one_charge(charge, dist, result, vs):
    assert vs._monopole_esp_one_charge(charge, dist) == result


def test_monopole_esp_one_charge_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs._monopole_esp_one_charge(1, 0)


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
    assert vs._monopole_esp_two_charges(charge1, charge2, dist1, dist2) == result


def test_monopole_esp_two_charges_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs._monopole_esp_two_charges(1, 1, 0, 0)


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
        vs._monopole_esp_three_charges(charge1, charge2, charge3, dist1, dist2, dist3)
        == result
    )


def test_monopole_esp_three_charges_div_zero(vs):
    with pytest.raises(ZeroDivisionError):
        vs._monopole_esp_three_charges(1, 1, 1, 0, 0, 0)


def test_dipole_esp(mol, vs):
    # convert from atomic units
    dipole_moment = mol.atoms[1].dipole.to_array() * BOHR_TO_ANGS

    assert vs._dipole_esp(np.array([1, 1, 1]), dipole_moment, 1) == pytest.approx(
        -1.76995515193e-29
    )


def test_quadrupole_esp(mol, vs):
    # convert from atomic units
    m_tensor = mol.atoms[1].quadrupole.to_array() * BOHR_TO_ANGS**2
    assert vs._quadrupole_esp(np.array([1, 1, 1]), m_tensor, 1) == pytest.approx(
        9.40851165275e-30
    )


def test_cloud_penetration(mol, vs):
    cloud_pen_data = mol.atoms[1].cloud_pen
    a, b = cloud_pen_data.a, cloud_pen_data.b
    b /= BOHR_TO_ANGS
    assert vs._cloud_penetration(a, b, 1) == pytest.approx(2.86224473231e-27)


def test_generate_sample_points_relative(vs):
    points = vs._generate_sample_points_relative(vdw_radius=1)
    for point in points:
        # All points should be 1.4-2.0x the vdw radius (in this instance, 1 Ang)
        assert 1.39 <= vs._xyz_distance(point, np.array([0, 0, 0])) <= 2.01


def test_get_vector_from_coords(vs, mol):
    # force them in as they are only cached when fitting sites
    vs._molecule = mol
    vs._coords = mol.coordinates
    vector = vs._get_vector_from_coords(atom_index=1, n_sites=1, alt=False)
    # Chlorine scale factor == 1.5
    assert np.linalg.norm(vector) == pytest.approx(1.5)
    vs._clear_cache()


def test_fit(mol, vs, tmpdir):
    with tmpdir.as_cwd():
        # make sure this is the reference value
        assert mol.atoms[1].aim.charge == -0.183627
        vs.run(molecule=mol)
        # make sure we have a site
        assert mol.extra_sites.n_sites == 2
        # make sure only the parent site has its charge changed
        assert mol.atoms[1].aim.charge != pytest.approx(
            float(mol.NonbondedForce[(1,)].charge)
        )
        # make sure the other values are similar to the aim values
        for atom in mol.atoms:
            if atom.atom_index != 1:
                assert atom.aim.charge == pytest.approx(
                    float(mol.NonbondedForce[(atom.atom_index,)].charge), abs=1e-5
                )
        lj = get_protocol(protocol_name="0")
        # add fake Cl param
        lj.free_parameters["Cl"] = cl_base(r_free=1.88)
        lj.run(molecule=mol)
        # make sure lJ did not reset the charge on the parent
        assert mol.atoms[1].aim.charge != pytest.approx(
            float(mol.NonbondedForce[(1,)].charge)
        )

        assert (
            sum(param.charge for param in mol.NonbondedForce)
            + sum(site.charge for site in mol.extra_sites)
            == 0
        )


def test_refit(mol, vs, tmpdir):
    """Make sure restarting a vsite fit produces the same result and removes old sites."""
    with tmpdir.as_cwd():
        vs.run(molecule=mol)
        assert mol.extra_sites.n_sites == 2
        # the new ref value
        ref = mol.copy(deep=True)
        vs.run(molecule=mol)
        # make sure old sites are removed
        assert mol.extra_sites.n_sites == 2
        # check the sites are the same
        assert ref.extra_sites[1][0].charge == mol.extra_sites[1][0].charge
        assert ref.extra_sites[1][1].charge == mol.extra_sites[1][1].charge
        assert ref.NonbondedForce[(1,)].charge == mol.NonbondedForce[(1,)].charge
        # make sure the aim data was not changed
        assert ref.atoms[1].aim.charge == mol.atoms[1].aim.charge


def test_vsite_frozen_angles(methanol, vs, tmpdir):
    """
    Test fitting vsites with the angle between 2 sites frozen
    """

    with tmpdir.as_cwd():
        vs.freeze_site_angles = True
        vs.run(molecule=methanol)
        assert methanol.extra_sites.n_sites == 2

        # check the angle between the sites is 90 degrees
        sites = []
        center_atom = None
        with open("xyz_with_extra_point_charges.xyz") as xyz:
            for line in xyz.readlines():
                if line.startswith("X"):
                    sites.append(np.array([float(x) for x in line.split()[1:4]]))
                elif line.startswith("O"):
                    center_atom = np.array([float(x) for x in line.split()[1:4]])
        # work out the angle
        b1, b2 = sites[0] - center_atom, sites[1] - center_atom
        cosine_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
        assert pytest.approx(90) == np.degrees(np.arccos(cosine_angle))


def test_vsite_opt_angles(methanol, vs, tmpdir):
    """
    Test fitting vsites with the angle between 2 sites frozen
    """

    with tmpdir.as_cwd():
        vs.freeze_site_angles = False
        vs.run(molecule=methanol)
        assert methanol.extra_sites.n_sites == 2

        # check the angle between the sites is 90 degrees
        sites = []
        center_atom = None
        with open("xyz_with_extra_point_charges.xyz") as xyz:
            for line in xyz.readlines():
                if line.startswith("X"):
                    sites.append(np.array([float(x) for x in line.split()[1:4]]))
                elif line.startswith("O"):
                    center_atom = np.array([float(x) for x in line.split()[1:4]])
        # work out the angle
        b1, b2 = sites[0] - center_atom, sites[1] - center_atom
        cosine_angle = np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))
        assert pytest.approx(110.1, abs=0.1) == np.degrees(np.arccos(cosine_angle))
