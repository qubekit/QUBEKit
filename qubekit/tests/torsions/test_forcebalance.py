"""
Forcebalance fitting specific tests.
"""

import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from qubekit.molecules import Ligand, TorsionDriveData
from qubekit.parametrisation import OpenFF
from qubekit.torsions import ForceBalanceFitting, Priors, TorsionProfile
from qubekit.utils.exceptions import (
    ForceBalanceError,
    MissingReferenceData,
    TorsionDriveDataError,
)
from qubekit.utils.file_handling import get_data
from qubekit.utils.helpers import export_torsiondrive_data


@pytest.fixture
def biphenyl():
    """
    Load up a biphenyl molecule with some torsiondrive data.
    """
    # load biphenyl
    mol = Ligand.from_file(file_name=get_data("biphenyl.sdf"))
    # load the torsiondrive data
    td_data = TorsionDriveData.from_qdata(
        qdata_file=get_data("biphenyl_qdata.txt"), dihedral=(6, 10, 11, 8)
    )
    mol.add_qm_scan(scan_data=td_data)
    return mol


def test_prior_dict():
    """
    Make sure the prior is correctly formated for forcebalance.
    """
    priors = Priors(Proper_k=4)
    data = priors.format_priors()
    assert data["Proper/k"] == 4


def test_torsion_target_generation(tmpdir, biphenyl):
    """
    Make sure that we can make a target from a molecule with its torsion scan data.
    """
    with tmpdir.as_cwd():
        torsion_target = TorsionProfile()
        target_folders = torsion_target.prep_for_fitting(molecule=biphenyl)
        # the name we expect for the target folder
        target_name = "TorsionProfile_OpenMM_10_11"
        # now make sure the folders have been made
        assert len(target_folders) == 1
        assert target_folders[0] == target_name
        assert target_name in os.listdir()
        # now we need to check the forcebalance target files have been made
        required_files = ["molecule.pdb", "metadata.json", "qdata.txt", "scan.xyz"]
        for f in required_files:
            assert f in os.listdir(target_name)


def test_target_prep_no_data(tmpdir, biphenyl):
    """
    Make sure an error is raised if we try and prep the target with misssing data.
    """
    with tmpdir.as_cwd():
        torsion_target = TorsionProfile()
        biphenyl.qm_scans = []
        with pytest.raises(MissingReferenceData):
            torsion_target.prep_for_fitting(molecule=biphenyl)


def test_torsion_metadata(tmpdir, biphenyl):
    """
    Make sure that the metadata.json has the correct torsion index and torsion grid values.
    """
    with tmpdir.as_cwd():
        torsion_target = TorsionProfile()
        torsion_target.make_metadata(torsiondrive_data=biphenyl.qm_scans[0])
        with open("metadata.json") as data:
            metadata = json.load(data)
            assert metadata["dihedrals"] == [
                [6, 10, 11, 8],
            ]
            assert metadata["grid_spacing"] == [
                15,
            ]
            assert metadata["dihedral_ranges"] == [
                [-165, 180],
            ]


def test_forcebalance_add_target():
    """
    Make sure the forcebalance optimiser can add targets when needed.
    """
    fb = ForceBalanceFitting()
    fb.targets = {}
    torsion_target = TorsionProfile()
    fb.add_target(target=torsion_target)
    assert torsion_target.target_name in fb.targets


def test_generate_forcefield(tmpdir, biphenyl):
    """
    Test generating a fitting forcefield for forcebalance with the correct torsion terms tagged.
    """
    with tmpdir.as_cwd():
        os.mkdir("forcefield")
        # get some openff params for the ligand
        OpenFF().run(molecule=biphenyl)
        ForceBalanceFitting.generate_forcefield(molecule=biphenyl)
        # load the forcefield and check for cosmetic tags
        root = ET.parse(os.path.join("forcefield", "bespoke.xml")).getroot()
        for torsion in root.iter(tag="Proper"):
            a1 = torsion.get(key="class1")
            a2 = torsion.get(key="class2")
            a3 = torsion.get(key="class3")
            a4 = torsion.get(key="class4")
            atoms = [a1, a2, a3, a4]
            dihedral = [int(re.search("[0-9]+", atom).group()) for atom in atoms]
            # if we have the same central bond make sure we have the parametrise tag
            tag = torsion.get(key="parameterize", default=None)
            central_bond = dihedral[1:3]
            if central_bond == [10, 11] or central_bond == [11, 10]:
                # make sure we have a tag
                for t in tag.split(","):
                    assert t.strip() in ["k1", "k2", "k3", "k4"]
            else:
                assert tag is None


def test_generate_optimise_in(tmpdir, biphenyl):
    """
    Test generating an optimize in file which captures the correct forcebalance run time settings.
    """
    with tmpdir.as_cwd():
        # parametrise the molecule
        OpenFF().run(biphenyl)
        # set some non-default values
        fb = ForceBalanceFitting(
            penalty_type="L2", max_iterations=100, minimum_trust_radius=10
        )
        tp = TorsionProfile(restrain_k=100)
        fb.add_target(target=tp)
        # now run the setup
        target_folders = tp.prep_for_fitting(molecule=biphenyl)
        fb.generate_optimise_in(target_data={tp.target_name: target_folders})
        # read the file back in
        with open("optimize.in") as opt_in:
            opt_data = opt_in.read()

        assert "penalty_type L2" in opt_data
        assert "maxstep 100" in opt_data
        assert "mintrust 10" in opt_data
        assert "restrain_k 100" in opt_data
        assert "type TorsionProfile_OpenMM" in opt_data


def test_full_optimise(tmpdir, biphenyl):
    """
    Test the forcebalance wrapper by doing a full optimise run for a simple molecule. Also check that the optimised results are saved.
    """
    with tmpdir.as_cwd():
        OpenFF().run(biphenyl)
        # use default values
        fb = ForceBalanceFitting()
        fitted_molecule = fb.run(molecule=biphenyl)
        # now compare the fitted results
        for old_values in biphenyl.TorsionForce:
            central_bond = old_values.atoms[1:3]
            if central_bond == [10, 11] or central_bond == [11, 10]:
                # compare the k values, making sure they have been changed
                fitted_values = fitted_molecule.TorsionForce[old_values.atoms]
                assert fitted_values.k1 != old_values.k1
                assert fitted_values.k2 != old_values.k2
                assert fitted_values.k3 != old_values.k3
                assert fitted_values.k4 != old_values.k4


def test_optimise_no_target(tmpdir, biphenyl):
    """
    Make sure we raise an error if there is no target to optimise.
    """
    with tmpdir.as_cwd():
        fb = ForceBalanceFitting()
        fb.targets = {}
        with pytest.raises(ForceBalanceError):
            fb.run(molecule=biphenyl)


def test_check_torsion_drive_scan_range():
    """
    Make sure the scan range is always order lowest to largest.
    """
    td = TorsionDriveData(dihedral=(6, 10, 11, 8), torsion_drive_range=(10, -90))
    assert td.torsion_drive_range == (-90, 10)


def test_max_min_angle():
    """
    Make sure the max and min angle are updated.
    """
    td = TorsionDriveData(dihedral=(6, 10, 11, 8), torsion_drive_range=(-40, 40))
    assert td.max_angle == 40
    assert td.min_angle == -40


def test_from_qdata():
    """
    Make sure we can create a valid TorsionDriveData from qdata.
    """
    td = TorsionDriveData.from_qdata(
        dihedral=(6, 10, 11, 8), qdata_file=get_data("biphenyl_qdata.txt")
    )
    # validate all angles
    td.validate_angles()


def test_central_bond():
    """
    Check the central bond property.
    """
    td = TorsionDriveData(dihedral=(6, 10, 11, 8))
    assert td.central_bond == (10, 11)


def test_qdata_round_trip(tmpdir):
    """
    Try and round trip a qdata txt file to a TorsionDriveData object and back.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("biphenyl.sdf"))
        td_ref = TorsionDriveData.from_qdata(
            dihedral=(6, 10, 11, 8), qdata_file=get_data("biphenyl_qdata.txt")
        )
        # now write to file
        export_torsiondrive_data(molecule=mol, tdrive_data=td_ref)
        # now make a second object
        td_new = TorsionDriveData.from_qdata(
            dihedral=(6, 10, 11, 8), qdata_file="qdata.txt"
        )
        assert td_ref.dihedral == td_new.dihedral
        for angle, ref_result in td_ref.reference_data.items():
            new_result = td_new.reference_data[angle]
            assert ref_result.angle == new_result.angle
            assert ref_result.energy == pytest.approx(new_result.energy)
            assert np.allclose(
                ref_result.geometry.tolist(), new_result.geometry.tolist()
            )


def test_missing_reference_data():
    """
    Make sure that missing reference data is identified in a TorsionDriveData object.
    """
    td = TorsionDriveData.from_qdata(
        dihedral=(6, 10, 11, 8), qdata_file=get_data("biphenyl_qdata.txt")
    )
    del td.reference_data[0]
    with pytest.raises(MissingReferenceData):
        td.validate_angles()


def test_inconsistent_scan_range():
    """
    Make sure an error is raised if we try and make a torsiondrivedata object which is inconsistent.
    """
    with pytest.raises(TorsionDriveDataError):
        _ = TorsionDriveData.from_qdata(
            dihedral=(6, 10, 11, 8),
            qdata_file=get_data("biphenyl_qdata.txt"),
            grid_spacing=10,
        )
