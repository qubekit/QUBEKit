"""
Test generating charge reference data or storing charges.
"""

import os

import numpy as np
import pytest
from qcelemental.models import AtomicInput
from qcelemental.util import which_import
from qcengine.config import get_config

from qubekit.charges import DDECCharges, MBISCharges, SolventPsi4
from qubekit.engines import GaussianHarness
from qubekit.molecules import Ligand
from qubekit.parametrisation import OpenFF
from qubekit.utils import constants
from qubekit.utils.exceptions import SpecificationError
from qubekit.utils.file_handling import get_data


def test_mbis_water_no_symm(tmpdir):
    """
    Make sure we can generate some mbis values for a molecule in a water solvent.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("water.pdb"))
        OpenFF().parametrise_molecule(molecule=mol)
        charge_method = MBISCharges(
            apply_symmetry=False,
            basis="sto-3g",
            method="hf",
            cores=1,
            memory=1,
            solvent_settings=SolventPsi4(medium_Solvent="water", units="au"),
        )
        mol = charge_method.run(molecule=mol)
        assert mol.atoms[0].aim.charge == pytest.approx(-0.60645602)
        assert mol.atoms[1].aim.charge == pytest.approx(0.30312772)
        assert np.allclose(
            mol.atoms[0].dipole.to_array(),
            np.array([-0.00340732, -0.02118032, -0.00321348]),
        )
        assert np.allclose(
            mol.atoms[1].dipole.to_array(),
            np.array([0.00601651, -0.00138073, -0.00013425]),
        )
        assert np.allclose(
            mol.atoms[0].quadrupole.to_array(),
            np.array(
                [
                    [0.02579991, 0.00198263, 0.00139757],
                    [0.00198263, 0.03733655, 0.01542122],
                    [0.00139757, 0.01542122, -0.06313645],
                ]
            ),
        )
        assert np.allclose(
            mol.atoms[1].quadrupole.to_array(),
            np.array(
                [
                    [-8.38231138e-03, 1.54368625e-02, 2.10736120e-03],
                    [1.54368625e-02, -7.36342859e-05, -1.11254560e-03],
                    [2.10736120e-03, -1.11254560e-03, 8.45594566e-03],
                ]
            ),
        )


def test_mbis_water_symm(tmpdir):
    """
    Make sure symmetry is correctly applied when requested to the reference charge values.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("water.pdb"))
        OpenFF().parametrise_molecule(molecule=mol)
        charge_method = MBISCharges(
            apply_symmetry=True,
            basis="sto-3g",
            method="hf",
            cores=1,
            memory=1,
            solvent_settings=SolventPsi4(medium_Solvent="water", units="au"),
        )
        mol = charge_method.run(molecule=mol)
        assert mol.atoms[1].aim.charge == float(mol.atoms[2].aim.charge)
        assert mol.atoms[1].aim.volume == mol.atoms[2].aim.volume
        # make sure the force is updated as well
        assert mol.atoms[0].aim.charge == float(mol.NonbondedForce[(0,)].charge)
        assert mol.NonbondedForce[(1,)].charge == mol.NonbondedForce[(2,)].charge
        # make sure the quadrupole is traceless
        for atom in mol.atoms:
            assert np.trace(atom.quadrupole.to_array()) == pytest.approx(0)


def test_mbis_available():
    """MBIS should always be available with the latest version of psi4."""
    if which_import("psi4", return_bool=True, raise_error=False):
        assert MBISCharges.is_available() is True
    else:
        with pytest.raises(ModuleNotFoundError):
            MBISCharges.is_available()


def test_pcm_spec_error():
    """
    Make sure an error is raised if we try and request an invalid solvent from pcm.
    """
    with pytest.raises(SpecificationError):
        _ = SolventPsi4(units="au", medium_Solvent="bad_solvent")


def test_pcm_unit_conversion():
    """Make sure the pcm units are correctly converted from defaults."""
    spec = SolventPsi4(units="angstrom", medium_Solvent="water")
    assert spec.cavity_Area == 0.3 * constants.BOHR_TO_ANGS ** 2


def test_ddec_available():
    """
    DDEC should only be available if the user has gaussian and chargemol.
    Note this is not possible for the CI so we expect it to fail here.
    """
    with pytest.raises(RuntimeError):
        DDECCharges.is_available()


@pytest.mark.parametrize(
    "version", [pytest.param(3, id="DDEC3"), pytest.param(6, id="DDEC6")]
)
def test_chargemol_template(tmpdir, version):
    """
    Make sure we can correctly render a chargemol template job.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("water.pdb"))
        OpenFF().parametrise_molecule(molecule=mol)
        charge_method = DDECCharges(
            apply_symmetry=True,
            basis="sto-3g",
            method="hf",
            cores=1,
            memory=1,
            ddec_version=version,
        )
        # fake the chargemol dir
        os.environ["CHARGEMOL_DIR"] = "test"
        # now render the template
        charge_method._build_chargemol_input(density_file_name="test.wfx", molecule=mol)
        with open("job_control.txt") as job_file:
            job_data = job_file.readlines()

        assert f"DDEC{version}\n" in job_data
        assert "test.wfx\n" in job_data
        assert "test/atomic_densities/\n" in job_data
        assert f"{mol.charge}\n" in job_data


def test_gaussian_solvent_template():
    """
    Make sure that the template rendered with solvent settings matches what we expect.
    """
    mol = Ligand.from_file(get_data("water.pdb"))
    # get the charge method and implicit solvent engine
    charge_engine = DDECCharges()
    solvent_settings = charge_engine.solvent_settings.format_keywords()
    # now make an atomic input for the harness
    task = AtomicInput(
        molecule=mol.to_qcschema(),
        driver="energy",
        model={"method": "b3lyp-d3bj", "basis": "6-311G"},
        keywords=solvent_settings,
    )
    # we need the harness as this will render the template
    gaussian_harness = GaussianHarness()
    config = get_config(local_options={"ncores": 1, "memory": 1})
    job_inputs = gaussian_harness.build_input(task, config)
    # make sure the job file matches or expected reference
    with open(get_data("gaussian_solvent_example.com")) as g_out:
        assert g_out.read() == job_inputs["infiles"]["gaussian.com"]
