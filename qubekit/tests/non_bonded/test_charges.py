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
            np.array([-2.14226675e-02, 3.40756266e-03, 1.40326120e-15]),
        )
        assert np.allclose(
            mol.atoms[1].dipole.to_array(),
            np.array([-1.38538356e-03, -6.01694191e-03, 8.59079094e-16]),
        )
        assert np.allclose(
            mol.atoms[0].quadrupole.to_array(),
            np.array(
                [
                    [-3.73567748e00, -2.16793592e-03, -1.49239507e-15],
                    [-2.16793592e-03, -3.74951479e00, -1.02141657e-15],
                    [-1.49239507e-15, -1.02141657e-15, -3.84079078e00],
                ]
            ),
        )
        assert np.allclose(
            mol.atoms[1].quadrupole.to_array(),
            np.array(
                [
                    [-3.53716471e-01, -1.55798333e-02, 4.67831761e-17],
                    [-1.55798333e-02, -3.61888757e-01, -4.81285287e-17],
                    [4.67831761e-17, -4.81285287e-17, -3.44906862e-01],
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
        assert mol.atoms[1].aim.charge == mol.atoms[2].aim.charge
        assert mol.atoms[1].aim.volume == mol.atoms[2].aim.volume
        # make sure the force is updated as well
        assert mol.atoms[0].aim.charge == mol.NonbondedForce[(0,)].charge
        assert mol.NonbondedForce[(1,)].charge == mol.NonbondedForce[(2,)].charge


def test_mbis_available():
    """MBIS should always be available with the latest version of psi4."""
    if which_import("psi4", return_bool=True, raise_error=False):
        assert MBISCharges.is_available() is True
    else:
        with pytest.raises(ModuleNotFoundError):
            MBISCharges.is_available()


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
