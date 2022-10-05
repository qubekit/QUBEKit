"""
Forcebalance fitting specific tests.
"""

import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField

from qubekit.molecules import Ligand, TorsionDriveData
from qubekit.parametrisation import OpenFF
from qubekit.torsions import ForceBalanceFitting, Priors, TorsionProfileSmirnoff
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
        torsion_target = TorsionProfileSmirnoff()
        target_folders = torsion_target.prep_for_fitting(molecule=biphenyl)
        # the name we expect for the target folder
        target_name = "TorsionProfile_SMIRNOFF_biphenyl_10_11"
        # now make sure the folders have been made
        assert len(target_folders) == 1
        assert target_folders[0] == target_name
        assert target_name in os.listdir()
        # now we need to check the forcebalance target files have been made
        required_files = [
            "molecule.pdb",
            "metadata.json",
            "qdata.txt",
            "scan.xyz",
            "molecule.sdf",
        ]
        for f in required_files:
            assert f in os.listdir(target_name)


def test_target_prep_no_data(tmpdir, biphenyl):
    """
    Make sure an error is raised if we try and prep the target with misssing data.
    """
    with tmpdir.as_cwd():
        torsion_target = TorsionProfileSmirnoff()
        biphenyl.qm_scans = []
        with pytest.raises(MissingReferenceData):
            torsion_target.prep_for_fitting(molecule=biphenyl)


def test_torsion_metadata(tmpdir, biphenyl):
    """
    Make sure that the metadata.json has the correct torsion index and torsion grid values.
    """
    with tmpdir.as_cwd():
        torsion_target = TorsionProfileSmirnoff()
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
    torsion_target = TorsionProfileSmirnoff()
    fb.add_target(target=torsion_target)
    assert torsion_target.target_name in fb.targets


def test_generate_forcefield(tmpdir, biphenyl):
    """
    Test generating a fitting forcefield for forcebalance with the correct torsion terms tagged.
    """
    with tmpdir.as_cwd():
        fb = ForceBalanceFitting()
        os.mkdir("forcefield")
        # get some openff params for the ligand
        OpenFF().run(molecule=biphenyl)
        fb.generate_forcefield(molecule=biphenyl)
        # load the forcefield and check for cosmetic tags
        offxml = ForceField(
            os.path.join("forcefield", "bespoke.offxml"),
            load_plugins=True,
            allow_cosmetic_attributes=True,
        )
        torsions = offxml.get_parameter_handler("ProperTorsions")
        for parameter in torsions.parameters:
            if parameter.attribute_is_cosmetic("parameterize"):
                p_tags = parameter._parameterize
                for t in p_tags.split(","):
                    assert t.strip() in ["k1", "k2", "k3", "k4"]
                # check the parameter matches the central rotatable bond
                matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
                for match in matches:
                    assert tuple(sorted(match[1:3])) == (10, 11)


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
        tp = TorsionProfileSmirnoff(restrain_k=100)
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
        assert "type TorsionProfile_SMIRNOFF" in opt_data


def test_full_optimise(tmpdir, biphenyl):
    """
    Test the forcebalance wrapper by doing a full optimise run for a simple molecule.
    Also check that the optimised results are saved.
    """
    with tmpdir.as_cwd():
        OpenFF().run(biphenyl)
        # use default values
        fb = ForceBalanceFitting()
        fit_molecule = biphenyl.copy(deep=True)
        fit_molecule = fb.run(molecule=fit_molecule)
        # now compare the fitted results
        master_terms = None
        for old_values in biphenyl.TorsionForce:
            central_bond = old_values.atoms[1:3]
            if central_bond == (10, 11) or central_bond == (11, 10):
                # compare the k values, making sure they have been changed
                fitted_values = fit_molecule.TorsionForce[old_values.atoms]
                if master_terms is None:
                    master_terms = fitted_values.copy(deep=True)
                assert fitted_values.k1 != old_values.k1
                assert fitted_values.k2 != old_values.k2
                assert fitted_values.k3 != old_values.k3
                assert fitted_values.k4 != old_values.k4
                # make sure all terms are the same between the symmetry equivalent dihedrals
                assert fitted_values.k1 == master_terms.k1
                assert fitted_values.k2 == master_terms.k2
                assert fitted_values.k3 == master_terms.k3
                assert fitted_values.k4 == master_terms.k4


def test_full_optimise_fragment(tmpdir, biphenyl_fragments):
    """
    Test a full fb optimisation using a fragmented bipheynl system, and make sure the parameters of the
    parent are correctly updated.
    """

    with tmpdir.as_cwd():
        fb = ForceBalanceFitting()
        fit_molecule = biphenyl_fragments.copy(deep=True)
        fit_molecule = fb.run(molecule=fit_molecule)
        # find the rotatable bond in the parent molecule
        fragment = biphenyl_fragments.fragments[0]
        fragment_map = [
            fragment.atoms[i].map_index for i in fragment.qm_scans[0].central_bond
        ]
        central_bond = tuple(
            sorted(
                [
                    biphenyl_fragments.get_atom_with_map_index(map_id).atom_index
                    for map_id in fragment_map
                ]
            )
        )
        # compare the resulting parameters
        for dihedrals in fit_molecule.dihedral_types.values():
            unique_torsions = set()
            # loop over the symmetry groups and make sure they have the same torsion parameters
            for dihedral in dihedrals:
                new_parameter = fit_molecule.TorsionForce[dihedral]
                unique_torsions.add(new_parameter.json(exclude={"atoms"}))
                old_parameter = biphenyl_fragments.TorsionForce[dihedral]
                # check if the torsion runs through the rotatable bond
                if tuple(sorted(dihedral[1:3])) == central_bond:
                    # make sure they have been optimised
                    assert new_parameter.k1 != old_parameter.k1
                    assert new_parameter.k2 != old_parameter.k2
                    assert new_parameter.k3 != old_parameter.k3
                    assert new_parameter.k4 != old_parameter.k4
                else:
                    # make sure the old parameters are the same
                    assert new_parameter.k1 == old_parameter.k1
                    assert new_parameter.k2 == old_parameter.k2
                    assert new_parameter.k3 == old_parameter.k3
                    assert new_parameter.k4 == old_parameter.k4

            assert len(unique_torsions) == 1


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
