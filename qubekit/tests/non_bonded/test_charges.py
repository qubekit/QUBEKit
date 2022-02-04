"""
Test generating charge reference data or storing charges.
"""

import pytest
from qcelemental.models import AtomicInput
from qcelemental.util import which_import
from qcengine.config import get_config

from qubekit.charges import DDECCharges, MBISCharges, SolventPsi4
from qubekit.engines import GaussianHarness
from qubekit.parametrisation import OpenFF
from qubekit.utils import constants
from qubekit.utils.datastructures import LocalResource, QCOptions, TDSettings
from qubekit.utils.exceptions import SpecificationError
from qubekit.utils.file_handling import get_data

# def test_mbis_water_symm(tmpdir, water):
#     """
#     Make sure symmetry is correctly applied when requested to the reference charge values.
#     """
#     if not which_import("psi4", raise_error=False, return_bool=True):
#         pytest.skip("Skipping as PSI4 not installed.")
#     with tmpdir.as_cwd():
#         OpenFF().run(molecule=water)
#         charge_method = MBISCharges(
#             basis="sto-3g",
#             method="hf",
#             solvent_settings=SolventPsi4(medium_Solvent="water", units="au"),
#         )
#         local_options = LocalResource(cores=1, memory=1)
#         mol = charge_method.run(molecule=water, local_options=local_options)
#         # use approx as we do rounding with symmetry
#         assert mol.atoms[1].aim.charge == pytest.approx(float(mol.atoms[2].aim.charge))
#         assert mol.atoms[1].aim.volume == pytest.approx(mol.atoms[2].aim.volume)
#         # make sure the force is updated as well
#         assert mol.atoms[0].aim.charge == pytest.approx(
#             float(mol.NonbondedForce[(0,)].charge)
#         )
#         assert mol.NonbondedForce[(1,)].charge == pytest.approx(
#             mol.NonbondedForce[(2,)].charge
#         )
#         # make sure the quadrupole is traceless
#         for atom in mol.atoms:
#             assert np.trace(atom.quadrupole.to_array()) == pytest.approx(0)


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
    assert spec.cavity_Area == 0.3 * constants.BOHR_TO_ANGS**2


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
def test_chargemol_template(tmpdir, version, water):
    """
    Make sure we can correctly render a chargemol template job.
    """
    with tmpdir.as_cwd():
        OpenFF().run(molecule=water)
        charge_method = DDECCharges(
            ddec_version=version,
        )
        # now render the template
        charge_method._build_chargemol_input(
            density_file_name="test.wfx", molecule=water
        )
        with open("job_control.txt") as job_file:
            job_data = job_file.readlines()

        assert f"DDEC{version}\n" in job_data
        assert "test.wfx\n" in job_data
        assert f"{water.charge}\n" in job_data


@pytest.mark.parametrize(
    "version", [pytest.param(3, id="DDEC3"), pytest.param(6, id="DDEC6")]
)
def test_chargemol_run(tmpdir, water, version):
    """
    test running a chargemol calculation and storing the aim data.
    """

    wfx_file = get_data("water.wfx")
    with tmpdir.as_cwd():
        assert water.atoms[0].aim.charge is None
        charge_method = DDECCharges(ddec_version=version)
        water = charge_method._call_chargemol(
            density_file_content=open(wfx_file).read(), molecule=water, cores=1
        )

        assert water.atoms[0].aim.charge is not None


def test_gaussian_solvent_template(tmpdir, water):
    """
    Make sure that the template rendered with solvent settings matches what we expect.
    """
    with tmpdir.as_cwd():
        # get the charge method and implicit solvent engine
        charge_engine = DDECCharges()
        solvent_settings = charge_engine._get_calculation_settings()
        # now make an atomic input for the harness
        task = AtomicInput(
            molecule=water.to_qcschema(),
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


def test_gaussian_no_solvent_template(tmpdir, water):
    """
    Make sure that we can calculate the electron density with no implicit solvent.
    """
    with tmpdir.as_cwd():
        # get the charge method and implicit solvent engine
        charge_engine = DDECCharges(solvent_settings=None)
        settings = charge_engine._get_calculation_settings()
        task = AtomicInput(
            molecule=water.to_qcschema(),
            driver="energy",
            model={"method": "b3lyp-d3bj", "basis": "6-311G"},
            keywords=settings,
        )
        # we need the harness as this will render the template
        gaussian_harness = GaussianHarness()
        config = get_config(local_options={"ncores": 1, "memory": 1})
        job_inputs = gaussian_harness.build_input(task, config)
        with open(get_data("gaussian_gas_example.com")) as g_out:
            assert g_out.read() == job_inputs["infiles"]["gaussian.com"]


def test_gaussian_td_solvent_template(tmpdir, water):
    """
    Make sure that we can calculate the electron density with implicit solvent in a td-scf calculation.
    """
    with tmpdir.as_cwd():
        # get the charge method and implicit solvent engine
        charge_engine = DDECCharges()
        charge_engine.solvent_settings.solver_type = "IPCM"
        qc_spec = QCOptions(
            method="cam-b3lyp",
            basis="6-31G",
            program="gaussian",
            td_settings=TDSettings(),
        )
        options = LocalResource(cores=1, memory=1)
        with pytest.raises(SpecificationError):
            charge_engine.run(water, qc_spec=qc_spec, local_options=options)

        # as we can not run gaussian just make sure the solver was changed when we use td-scf
        assert charge_engine.solvent_settings.solver_type == "PCM"
