import openmm
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import unit

from qubekit.cli.combine import _add_water_model
from qubekit.cli.utils import LocalVirtualSite
from qubekit.molecules import Ligand


def test_local_site_to_dict():
    """Make sure our plugin vsite can be wrote to dict"""

    vsite = LocalVirtualSite(
        name="test",
        orientations=[(0, 1, 2)],
        p1=0 * unit.angstrom,
        p2=0 * unit.angstrom,
        p3=0 * unit.angstrom,
        o_weights=[1.0, 0.0, 0.0],
        x_weights=[-1.0, 0.5, 0.5],
        y_weights=[-1.0, 0.0, 1.0],
    )
    vsite_dict = vsite.to_dict()
    assert vsite_dict["name"] == "test"
    assert vsite_dict["p1"] == 0 * unit.nanometer
    assert vsite_dict["o_weights"] == [1.0, 0.0, 0.0]
    assert vsite_dict["x_weights"] == [-1.0, 0.5, 0.5]


def test_local_site_from_dict_fail():
    """Make sure our plugin site correctly raises an error when we try and make it from the wrong info"""

    with pytest.raises(AssertionError):
        LocalVirtualSite.from_dict({"vsite_type": "BondChargeVirtualSite"})


def test_local_site_from_dict():
    """Make sure our plugin vsite can be correctly loaded from a dict of data."""

    o_weights = [1.0, 0.0, 0.0]
    x_weights = [-1.0, 1.0, 0.0]
    y_weights = [-1.0, 0.0, 1.0]
    vsite_dict = {
        "name": "LP_from_dict",
        "vsite_type": "LocalVirtualSite",
        "p1": 1 * unit.angstrom,
        "p2": 0 * unit.angstrom,
        "p3": 1 * unit.angstrom,
        "orientations": [(0, 1, 2)],
        "o_weights": o_weights,
        "x_weights": x_weights,
        "y_weights": y_weights,
    }
    lp = LocalVirtualSite.from_dict(vsite_dict=vsite_dict)
    ref_lp = LocalVirtualSite(
        name="LP_from_dict",
        orientations=[(0, 1, 2)],
        p1=0.1 * unit.nanometer,
        p2=0 * unit.nanometer,
        p3=0.1 * unit.nanometer,
        o_weights=o_weights,
        x_weights=x_weights,
        y_weights=y_weights,
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
        o_weights=[1.0, 0.0, 0.0],
        x_weights=[-1.0, 1.0, 0.0],
        y_weights=[-1.0, 0.0, 1.0],
    )
    assert lp.p1 == 0.1 * unit.nanometer
    assert lp.p2 == 0.1 * unit.nanometer
    assert lp.p3 == 0 * unit.nanometer
    assert lp.local_frame_position == unit.Quantity([0.1, 0.1, 0], unit=unit.nanometer)


def test_local_get_openmm_site():
    """
    Make sure we can correctly convert our site to an openmm object.
    """
    lp = LocalVirtualSite(
        name="lp",
        orientations=[(0, 1, 2)],
        p1=1 * unit.angstrom,
        p2=1 * unit.angstrom,
        p3=2 * unit.angstrom,
        o_weights=[1.0, 0.0, 0.0],
        x_weights=[-1.0, 1.0, 0.0],
        y_weights=[-1.0, 0.0, 1.0],
    )
    openmm_site = lp.get_openmm_virtual_site(atoms=(1, 2, 3))
    openmm_position = openmm_site.getLocalPosition()
    assert openmm_position.x == lp.p1.value_in_unit(unit.nanometer)
    assert openmm_position.y == lp.p2.value_in_unit(unit.nanometer)
    assert openmm_position.z == lp.p3.value_in_unit(unit.nanometer)


@pytest.mark.parametrize(
    "smirks, correction, direction",
    [
        pytest.param("[#6!a:1]-[#8X2H1:2]", 0.1, "forward", id="Normal"),
        pytest.param("[#6!a:2]-[#8X2H1:1]", 0.1, "reverse", id="Reversed"),
        pytest.param("[#6!a:1]-[#8X2H1:2]", -0.1, "forward", id="Negative"),
        pytest.param(
            "[#6!a:2]-[#8X2H1:1]", -0.1, "reverse", id="Reversed and negative"
        ),
    ],
)
def test_bcc_correction(methanol, smirks, correction, direction):
    """
    Make sure that BCCs are correctly applied to openmm systems.

    Reverse just tells the test if the smirks have been flipped so the correction should be added to the first atom
    and subtracted from the other.
    """
    # remove the v-sites
    methanol.extra_sites.clear_sites()
    for i in range(methanol.n_atoms):
        methanol.NonbondedForce[(i,)].charge = methanol.atoms[i].aim.charge

    ff = ForceField(load_plugins=True)
    methanol.add_params_to_offxml(offxml=ff)
    # add an alcohol bcc
    bcc = ff.get_parameter_handler("BondChargeCorrection")
    bcc.add_parameter(
        {
            "smirks": smirks,  # charge to be transferred from C->O
            "charge_correction": correction * unit.elementary_charge,
        }
    )

    off_mol = Molecule.from_rdkit(methanol.to_rdkit())
    # create the system and make sure the BCC was applied
    openmm_system = ff.create_openmm_system(off_mol.to_topology())
    forces = dict(
        (force.__class__.__name__, force) for force in openmm_system.getForces()
    )
    nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]

    if direction == "reverse":
        correction *= -1

    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        qb_params = methanol.NonbondedForce[(i,)]
        atomic_number = methanol.atoms[i].atomic_number
        if atomic_number == 6:
            assert (
                charge.value_in_unit(unit.elementary_charge)
                == float(qb_params.charge) + correction
            )
        elif atomic_number == 8:
            assert (
                charge.value_in_unit(unit.elementary_charge)
                == float(qb_params.charge) - correction
            )
        else:
            assert charge.value_in_unit(unit.elementary_charge) == float(
                qb_params.charge
            )

        # make sure sigma and epsilon are the same
        assert sigma.value_in_unit(unit.nanometers) == qb_params.sigma
        assert epsilon.value_in_unit(unit.kilojoule_per_mole) == qb_params.epsilon


def test_multi_bbc_correction(openff, tmpdir):
    """
    For a molecule which can match many types of correction make sure they are applied to a system that also has other
    particles including water with a v-site.
    """
    with tmpdir.as_cwd():
        water = Molecule.from_smiles("O")
        mol = Ligand.from_smiles("Fc1ccccc1", "fluorobenzene")
        openff.run(mol)
        ff = ForceField(load_plugins=True)
        mol.add_params_to_offxml(offxml=ff)
        _add_water_model(force_field=ff, water_model="tip4p-fb")
        bcc = ff.get_parameter_handler("BondChargeCorrection")
        bcc.add_parameter(
            {
                "smirks": "[#6a:1]-[#9:2]",
                "charge_correction": 0.1 * unit.elementary_charge,
            }
        )
        bcc.add_parameter(
            {
                "smirks": "[#1:1]-[#6a:2]",
                "charge_correction": 0.05 * unit.elementary_charge,
            }
        )

        off_mol = Molecule.from_rdkit(mol.to_rdkit())
        off_top = Topology.from_molecules(molecules=[off_mol, water])
        openmm_system = ff.create_openmm_system(off_top)
        forces = dict(
            (force.__class__.__name__, force) for force in openmm_system.getForces()
        )
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]

        for parameter in bcc.parameters:
            matches = off_mol.chemical_environment_matches(parameter.smirks)
            for match in matches:
                atom_add, atom_subtract = match
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_add)
                assert float(
                    mol.NonbondedForce[(atom_add,)].charge
                ) + parameter.charge_correction.value_in_unit(
                    unit.elementary_charge
                ) == charge.value_in_unit(
                    unit.elementary_charge
                )
