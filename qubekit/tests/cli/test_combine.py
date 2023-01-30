import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

import numpy as np
import pytest
import xmltodict
from deepdiff import DeepDiff
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import XmlSerializer, app, openmm, unit

import qubekit
from qubekit.charges import MBISCharges
from qubekit.cli.combine import (
    _add_water_model,
    _combine_molecules,
    _combine_molecules_offxml,
    _find_molecules_and_rfrees,
    _get_eval_string,
    _get_parameter_code,
    _update_increment,
    combine,
    elements,
)
from qubekit.molecules import Ligand
from qubekit.nonbonded import LennardJones612, get_protocol
from qubekit.utils import constants
from qubekit.utils.file_handling import get_data


@pytest.mark.parametrize(
    "molecule, atom_index, expected",
    [
        pytest.param("water", 0, "O", id="O"),
        pytest.param("water", 1, "X", id="Polar H"),
        pytest.param("acetone", 4, "H", id="Normal H"),
    ],
)
def test_get_param_code(molecule, atom_index, expected, request):
    """
    Test getting the rfree param code including polar h.
    """
    molecule = request.getfixturevalue(molecule)
    rfree_code = _get_parameter_code(molecule=molecule, atom_index=atom_index)
    assert rfree_code == expected


def test_mutual_args(run_cli):
    """
    Make sure an error is raised if we supply parameters and no-targets args.
    """
    result = run_cli.invoke(combine, args=["combined.xml", "-p", "C", "--no-targets"])
    assert result.exit_code == 1
    assert (
        "The options parameters and no-targets are mutually exclusive." in result.output
    )


def test_update_increment():
    """
    Make sure force data has the class incremented correctly.
    """
    force_data = {"class1": "1", "class2": "2", "k": "300"}
    new_data = _update_increment(force_data=force_data, increment=100)
    assert new_data["class1"] == "101"
    assert new_data["class2"] == "102"


def test_find_molecules_and_rfree(acetone, tmpdir, rdkit_workflow):
    """
    Test looping over folders to find rfree
    """

    with tmpdir.as_cwd():
        # make some example folders
        os.mkdir("QUBEKit_mol1")
        result = rdkit_workflow._build_initial_results(molecule=acetone)
        result.to_file(os.path.join("QUBEKit_mol1", "workflow_result.json"))

        molecules, rfree = _find_molecules_and_rfrees()
        assert len(molecules) == 1
        assert len(rfree) == 7


@pytest.mark.parametrize(
    "atom, a_and_b, rfree_code, expected",
    [
        pytest.param(
            1,
            True,
            "C",
            f"epsilon=(PARM['xalpha/alpha']*46.6*(21.129491/34.4)**PARM['xbeta/beta'])/(128*PARM['CElement/cfree']**6)*{constants.EPSILON_CONVERSION}, sigma=2**(5/6)*(21.129491/34.4)**(1/3)*PARM['CElement/cfree']*{constants.SIGMA_CONVERSION}",
            id="AB Rfree",
        ),
        pytest.param(
            1,
            False,
            "C",
            f"epsilon=(1.32*46.6*(21.129491/34.4)**0.469)/(128*PARM['CElement/cfree']**6)*{constants.EPSILON_CONVERSION}, sigma=2**(5/6)*(21.129491/34.4)**(1/3)*PARM['CElement/cfree']*{constants.SIGMA_CONVERSION}",
            id="No AB Rfree",
        ),
        pytest.param(
            1,
            True,
            None,
            f"epsilon=(PARM['xalpha/alpha']*46.6*(21.129491/34.4)**PARM['xbeta/beta'])/(128*2**6)*{constants.EPSILON_CONVERSION}, sigma=2**(5/6)*(21.129491/34.4)**(1/3)*2*{constants.SIGMA_CONVERSION}",
            id="AB No Rfree",
        ),
        pytest.param(
            1,
            False,
            None,
            f"epsilon=(1.32*46.6*(21.129491/34.4)**0.469)/(128*2**6)*{constants.EPSILON_CONVERSION}, sigma=2**(5/6)*(21.129491/34.4)**(1/3)*2*{constants.SIGMA_CONVERSION}",
            id="No AB No Rfree",
        ),
    ],
)
def test_get_eval_string(atom, a_and_b, rfree_code, expected, coumarin):
    """
    Test generating the eval string for ForceBalance with different settings.
    """
    rfree_data = {"v_free": 34.4, "b_free": 46.6, "r_free": 2}  # carbon

    eval_string = _get_eval_string(
        atom=coumarin.atoms[atom],
        rfree_data=rfree_data,
        a_and_b=a_and_b,
        rfree_code=rfree_code,
        alpha_ref="1.32",
        beta_ref="0.469",
    )
    assert eval_string == expected, print(eval_string)


def test_combine_molecules_deepdiff(acetone, openff, coumarin, tmpdir, rfree_data):

    with tmpdir.as_cwd():
        openff.run(acetone)
        acetone_ref_system = xmltodict.parse(open("serialised_acetone.xml").read())
        openff.run(coumarin)
        coumarin_ref_system = xmltodict.parse(open("serialised_coumarin.xml").read())

        combined_xml = _combine_molecules(
            molecules=[acetone, coumarin], parameters=elements, rfree_data=rfree_data
        ).getroot()
        messy = ET.tostring(combined_xml, "utf-8")

        pretty_xml = parseString(messy).toprettyxml(indent="")

        with open("combined.xml", "w") as xml_doc:
            xml_doc.write(pretty_xml)
        root = combined_xml.find("QUBEKit")
        assert qubekit.__version__ == root.get("Version")

        # load up new systems and compare
        combinded_ff = app.ForceField("combined.xml")

        acetone_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combinded_ff.createSystem(
                    acetone.to_openmm_topology(),
                    nonbondedCutoff=0.9,
                    removeCMMotion=False,
                )
            )
        )
        # slight differences in masses we need to ignore
        acetone_diff = DeepDiff(
            acetone_ref_system,
            acetone_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(acetone_diff) == 0

        coumarin_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combinded_ff.createSystem(
                    coumarin.to_openmm_topology(),
                    nonbondedCutoff=0.9,
                    removeCMMotion=False,
                )
            )
        )
        coumarin_diff = DeepDiff(
            coumarin_ref_system,
            coumarin_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(coumarin_diff) == 0


def test_combine_molecules_sites_deepdiff(openff, xml, acetone, rfree_data, tmpdir):
    """
    Test combining molecules with virtual sites and ensure they are correctly applied and the energy break down matches.
    """
    with tmpdir.as_cwd():
        openff.run(acetone)
        acetone_ref_system = xmltodict.parse(open("serialised_acetone.xml").read())
        pyridine = Ligand.from_file(file_name=get_data("pyridine.sdf"))
        xml.run(molecule=pyridine, input_files=[get_data("pyridine.xml")])
        pyridine_ref_system = xmltodict.parse(open("serialised_pyridine.xml").read())

        combined_xml = _combine_molecules(
            molecules=[acetone, pyridine], parameters=elements, rfree_data=rfree_data
        ).getroot()
        messy = ET.tostring(combined_xml, "utf-8")

        pretty_xml = parseString(messy).toprettyxml(indent="")
        with open("combined.xml", "w") as xml_doc:
            xml_doc.write(pretty_xml)
        root = combined_xml.find("QUBEKit")
        assert qubekit.__version__ == root.get("Version")

        # load up new systems and compare
        combinded_ff = app.ForceField("combined.xml")

        acetone_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combinded_ff.createSystem(
                    acetone.to_openmm_topology(),
                    nonbondedCutoff=0.9,
                    removeCMMotion=False,
                )
            )
        )
        # slight differences in masses we need to ignore
        acetone_diff = DeepDiff(
            acetone_ref_system,
            acetone_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(acetone_diff) == 0
        # add v-site
        pyridine_mod = app.Modeller(
            pyridine.to_openmm_topology(), pyridine.openmm_coordinates()
        )
        pyridine_mod.addExtraParticles(combinded_ff)
        pyridine_combine_system = xmltodict.parse(
            XmlSerializer.serialize(combinded_ff.createSystem(pyridine_mod.topology))
        )
        pyridine_diff = DeepDiff(
            pyridine_ref_system,
            pyridine_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(pyridine_diff) == 0


@pytest.mark.parametrize(
    "parameters, expected",
    [
        pytest.param(["-p", "C"], 1, id="Only C"),
        pytest.param(
            ["-p", "C", "-p", "AB", "-p", "N", "-p", "O"], 5, id="CNO alpha beta"
        ),
        pytest.param([], 7, id="All present"),
        pytest.param(["--no-targets"], 0, id="No targets"),
    ],
)
def test_combine_cli_all(
    run_cli, tmpdir, acetone, coumarin, openff, rdkit_workflow, parameters, expected
):
    """
    Test combining via the cli with different parameters.
    """
    workflow = rdkit_workflow.copy(deep=True)
    workflow.non_bonded = get_protocol(protocol_name="5b")
    with tmpdir.as_cwd():
        openff.run(coumarin)
        openff.run(acetone)
        # make some dummy dirs
        os.mkdir("QUBEKit_acetone")
        os.mkdir("QUBEKit_coumarin")
        result = workflow._build_initial_results(molecule=acetone)
        result.to_file(os.path.join("QUBEKit_acetone", "workflow_result.json"))
        result = workflow._build_initial_results(molecule=coumarin)
        result.to_file(os.path.join("QUBEKit_coumarin", "workflow_result.json"))
        output = run_cli.invoke(combine, args=["combined.xml", *parameters])
        assert output.exit_code == 0
        assert "2 molecules found, combining..." in output.output

        data = xmltodict.parse(open("combined.xml").read())
        if expected > 0:
            assert len(data["ForceField"]["ForceBalance"]) == expected
        else:
            assert data["ForceField"]["ForceBalance"] is None


@pytest.mark.parametrize(
    "parameters, expected",
    [
        pytest.param(["-p", "C"], 1, id="Only C"),
        pytest.param(
            ["-p", "C", "-p", "AB", "-p", "N", "-p", "O"], 5, id="CNO alpha beta"
        ),
        pytest.param([], 7, id="All present"),
        pytest.param(["--no-targets"], 0, id="No targets"),
    ],
)
def test_combine_cli_all_offxml(
    run_cli, tmpdir, acetone, coumarin, openff, rdkit_workflow, parameters, expected
):
    """
    Test combining offxmls via the cli with different parameters.
    """
    workflow = rdkit_workflow.copy(deep=True)
    workflow.non_bonded = get_protocol(protocol_name="5b")
    with tmpdir.as_cwd():
        openff.run(coumarin)
        openff.run(acetone)
        # make some dummy dirs
        os.mkdir("QUBEKit_acetone")
        os.mkdir("QUBEKit_coumarin")
        result = workflow._build_initial_results(molecule=acetone)
        result.to_file(os.path.join("QUBEKit_acetone", "workflow_result.json"))
        result = workflow._build_initial_results(molecule=coumarin)
        result.to_file(os.path.join("QUBEKit_coumarin", "workflow_result.json"))
        output = run_cli.invoke(
            combine, args=["combined.offxml", "-offxml", *parameters]
        )
        assert output.exit_code == 0
        assert "2 molecules found, combining..." in output.output

        if expected != 0:
            ff = ForceField(
                "combined.offxml", allow_cosmetic_attributes=True, load_plugins=True
            )
            # make sure we have used the plugin method
            vdw_handler = ff.get_parameter_handler("QUBEKitvdWTS")
            assert len(vdw_handler.parameters) == 32
            assert "parameterize" in vdw_handler._cosmetic_attribs
            assert len(getattr(vdw_handler, "_parameterize").split(",")) == expected
        else:
            # we are using the normal format so make sure it complies
            ff = ForceField("combined.offxml")
            vdw_handler = ff.get_parameter_handler("vdW")
            assert len(vdw_handler.parameters) == 32


@pytest.mark.parametrize(
    "molecule",
    [pytest.param("methanol", id="methanol"), pytest.param("ethanol", id="ethanol")],
)
def test_combine_sites_offxml_deepdiff(
    xml, tmpdir, rfree_data, methanol, ethanol, acetone, openff, molecule
):
    """
    Make sure we can create a combined offxml with vsites and get the same system with xml and offxml.
    Note this is not using the plugin.
    """

    molecules = {"methanol": methanol, "ethanol": ethanol}
    mol = molecules[molecule]

    with tmpdir.as_cwd():
        openff.run(acetone)
        off_acetone = Molecule.from_rdkit(acetone.to_rdkit())
        off_mol = Molecule.from_rdkit(mol.to_rdkit())
        acetone_ref_system = xmltodict.parse(open("serialised_acetone.xml").read())
        mol.write_parameters("mol.xml")
        mol_ff = app.ForceField("mol.xml")
        mol_mod = app.Modeller(mol.to_openmm_topology(), mol.openmm_coordinates())
        mol_mod.addExtraParticles(mol_ff)

        mol_system = mol_ff.createSystem(
            mol_mod.topology, nonbondedCutoff=0.9, removeCMMotion=False
        )
        mol_ref_system = xmltodict.parse(XmlSerializer.serialize(mol_system))
        # with open("methanol_ref.xml", "w") as out:
        #     out.write(XmlSerializer.serialize(methanol_system))

        _combine_molecules_offxml(
            molecules=[mol, acetone],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
            h_constraints=False,
        )

        combined_offml = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        acetone_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combined_offml.create_openmm_system(off_acetone.to_topology())
            )
        )
        acetone_diff = DeepDiff(
            acetone_ref_system,
            acetone_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(acetone_diff) == 1
        for item in acetone_diff["iterable_item_added"].values():
            assert item["@k"] == "0"

        mol_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combined_offml.create_openmm_system(off_mol.to_topology())
            )
        )
        # with open("methanol_comb.xml", "w") as out:
        #     out.write(XmlSerializer.serialize(combined_offml.create_openmm_system(off_methanol.to_topology())))

        mol_diff = DeepDiff(
            mol_ref_system,
            mol_combine_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(mol_diff) == 1
        for item in acetone_diff["iterable_item_added"].values():
            assert item["@k"] == "0"


def test_combine_rb_offxml(tmpdir, xml, rfree_data):
    """
    Make sure an error is raised if we try and make an offxml with RB terms.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("cyclohexane.sdf"))
        xml.run(molecule=mol, input_files=[get_data("cyclohexane.xml")])
        # there should be 6 RB terms
        assert mol.RBTorsionForce.n_parameters == 6
        # and no periodic terms
        assert mol.TorsionForce.n_parameters == 0
        # now try and make offxml
        _combine_molecules_offxml(
            molecules=[mol],
            parameters=elements,
            rfree_data=rfree_data,
            filename="test.offxml",
            h_constraints=False,
        )
        ff = ForceField(
            "test.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        rb_torsions = ff.get_parameter_handler("ProperRyckhaertBellemans")
        assert len(rb_torsions.parameters) == 1


def test_combine_molecules_deepdiff_offxml(
    acetone, openff, coumarin, tmpdir, rfree_data
):
    """When not optimising anything make sure we can round trip openff parameters"""

    with tmpdir.as_cwd():
        openff.run(acetone)
        acetone_ref_system = xmltodict.parse(open("serialised_acetone.xml").read())
        openff.run(coumarin)
        coumarin_ref_system = xmltodict.parse(open("serialised_coumarin.xml").read())

        _combine_molecules_offxml(
            molecules=[acetone, coumarin],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
            h_constraints=False,
        )

        # load up new systems and compare
        combinded_ff = ForceField("combined.offxml")
        assert combinded_ff.author == f"QUBEKit_version_{qubekit.__version__}"

        acetone_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combinded_ff.create_openmm_system(
                    topology=Molecule.from_rdkit(acetone.to_rdkit()).to_topology()
                )
            )
        )
        acetone_diff = DeepDiff(
            acetone_ref_system,
            acetone_combine_system,
            ignore_order=True,
            significant_digits=6,
        )
        # should only be a difference in torsions with zero k values as openff drops these
        assert len(acetone_diff) == 1
        for item in acetone_diff["iterable_item_added"].values():
            assert item["@k"] == "0"

        coumarin_combine_system = xmltodict.parse(
            XmlSerializer.serialize(
                combinded_ff.create_openmm_system(
                    topology=Molecule.from_rdkit(coumarin.to_rdkit()).to_topology()
                )
            )
        )
        coumarin_diff = DeepDiff(
            coumarin_ref_system,
            coumarin_combine_system,
            ignore_order=True,
            significant_digits=6,
        )
        assert len(coumarin_diff) == 1
        for item in coumarin_diff["iterable_item_added"].values():
            assert item["@k"] == "0"


@pytest.mark.parametrize(
    "parameters",
    [pytest.param([], id="No parameters"), pytest.param(elements, id="All parameters")],
)
def test_molecule_and_water_offxml(coumarin, water, tmpdir, rfree_data, parameters):
    """Test that an offxml can parameterize a molecule and water mixture."""

    coumarin_copy = coumarin.copy(deep=True)
    MBISCharges.apply_symmetrisation(coumarin_copy)

    with tmpdir.as_cwd():

        # run the lj method to make sure the parameters match
        alpha = rfree_data.pop("alpha")
        beta = rfree_data.pop("beta")
        lj = LennardJones612(free_parameters=rfree_data, alpha=alpha, beta=beta)
        # get new Rfree data
        lj.run(coumarin_copy)

        # remake rfree
        rfree_data["alpha"] = alpha
        rfree_data["beta"] = beta

        _combine_molecules_offxml(
            molecules=[coumarin_copy],
            parameters=parameters,
            rfree_data=rfree_data,
            filename="combined.offxml",
            water_model="tip3p",
            h_constraints=False,
        )

        combinded_ff = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(coumarin_copy.to_rdkit()),
            ]
        )
        system = combinded_ff.create_openmm_system(topology=mixed_top)
        # make sure we have 3 constraints
        assert system.getNumConstraints() == 3
        # check each constraint
        reference_constraints = [
            [0, 1, unit.Quantity(0.9572, unit=unit.angstroms)],
            [0, 2, unit.Quantity(0.9572, unit=unit.angstroms)],
            [1, 2, unit.Quantity(1.5139006545247014, unit=unit.angstroms)],
        ]
        for i in range(3):
            a, b, constraint = system.getConstraintParameters(i)
            assert a == reference_constraints[i][0]
            assert b == reference_constraints[i][1]
            assert constraint == reference_constraints[i][2].in_units_of(
                unit.nanometers
            )
        # now loop over the forces and make sure water was correctly parameterised
        forces = dict((force.__class__.__name__, force) for force in system.getForces())
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]
        # first check water has the correct parameters
        water_reference = [
            [
                unit.Quantity(-0.834, unit.elementary_charge),
                unit.Quantity(3.1507, unit=unit.angstroms),
                unit.Quantity(0.1521, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.417, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.417, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
        ]
        for i in range(3):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            assert charge == water_reference[i][0]
            assert sigma.in_units_of(unit.angstroms) == water_reference[i][1]
            assert (
                epsilon.in_units_of(unit.kilocalorie_per_mole) == water_reference[i][2]
            )

        # now check coumarin
        for i in range(coumarin_copy.n_atoms):
            ref_params = coumarin_copy.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i + 3)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon


def test_molecule_and_vsite_water(coumarin, tmpdir, water, rfree_data):
    """
    Make sure a v-site water model is correctly loaded from and offxml
    """
    coumarin_copy = coumarin.copy(deep=True)
    MBISCharges.apply_symmetrisation(coumarin_copy)

    with tmpdir.as_cwd():

        # run the lj method to make sure the parameters match
        alpha = rfree_data.pop("alpha")
        beta = rfree_data.pop("beta")
        lj = LennardJones612(free_parameters=rfree_data, alpha=alpha, beta=beta)
        # get new Rfree data
        lj.run(coumarin_copy)

        # remake rfree
        rfree_data["alpha"] = alpha
        rfree_data["beta"] = beta

        _combine_molecules_offxml(
            molecules=[coumarin_copy],
            parameters=elements,
            rfree_data=rfree_data,
            filename="combined.offxml",
            water_model="tip4p-fb",
            h_constraints=False,
        )

        combinded_ff = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        # make sure the sites are local and we have no normal sites
        handlers = combinded_ff.registered_parameter_handlers
        assert "LocalCoordinateVirtualSites" not in handlers
        assert "VirtualSites" in handlers
        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(coumarin_copy.to_rdkit()),
            ]
        )
        system = combinded_ff.create_openmm_system(topology=mixed_top)
        # make sure we have 3 constraints
        assert system.getNumConstraints() == 3
        # check each constraint
        reference_constraints = [
            [0, 1, unit.Quantity(0.9572, unit=unit.angstroms)],
            [0, 2, unit.Quantity(0.9572, unit=unit.angstroms)],
            [1, 2, unit.Quantity(1.5139006545247014, unit=unit.angstroms)],
        ]
        for i in range(3):
            a, b, constraint = system.getConstraintParameters(i)
            assert a == reference_constraints[i][0]
            assert b == reference_constraints[i][1]
            assert constraint == reference_constraints[i][2].in_units_of(
                unit.nanometers
            )
        # now loop over the forces and make sure water was correctly parameterised
        forces = dict((force.__class__.__name__, force) for force in system.getForces())
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]
        # first check water has the correct parameters
        water_reference = [
            [
                unit.Quantity(0.0, unit.elementary_charge),
                unit.Quantity(3.1655, unit=unit.angstroms),
                unit.Quantity(0.179082, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
        ]
        for i in range(3):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            assert charge == water_reference[i][0]
            assert sigma.in_units_of(unit.nanometer) == water_reference[i][
                1
            ].in_units_of(unit.nanometer)
            assert (
                epsilon.in_units_of(unit.kilocalorie_per_mole) == water_reference[i][2]
            )

        # now check coumarin
        for i in range(coumarin_copy.n_atoms):
            ref_params = coumarin_copy.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i + 3)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon

        # now check the vsite which should be last
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(
            coumarin_copy.n_atoms + 3
        )
        assert charge == unit.Quantity(-1.05174, unit.elementary_charge)
        assert sigma == unit.Quantity(1.0, unit.nanometer)
        assert epsilon == unit.Quantity(0.0, unit.kilojoule_per_mole)


def test_molecule_vsite_and_vsite_water(methanol, tmpdir, water, rfree_data):
    """
    Make sure a v-site water model is correctly loaded from and offxml
    """

    with tmpdir.as_cwd():

        _combine_molecules_offxml(
            molecules=[methanol],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
            water_model="tip4p-fb",
            h_constraints=False,
        )

        combinded_ff = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(methanol.to_rdkit()),
            ]
        )
        system = combinded_ff.create_openmm_system(topology=mixed_top)
        # make sure we have 3 constraints
        assert system.getNumConstraints() == 3
        # check each constraint
        reference_constraints = [
            [0, 1, unit.Quantity(0.9572, unit=unit.angstroms)],
            [0, 2, unit.Quantity(0.9572, unit=unit.angstroms)],
            [1, 2, unit.Quantity(1.5139006545247014, unit=unit.angstroms)],
        ]
        for i in range(3):
            a, b, constraint = system.getConstraintParameters(i)
            assert a == reference_constraints[i][0]
            assert b == reference_constraints[i][1]
            assert constraint == reference_constraints[i][2].in_units_of(
                unit.nanometers
            )
        # now loop over the forces and make sure water was correctly parameterised
        forces = dict((force.__class__.__name__, force) for force in system.getForces())
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]
        # first check water has the correct parameters
        water_reference = [
            [
                unit.Quantity(0.0, unit.elementary_charge),
                unit.Quantity(3.1655, unit=unit.angstroms),
                unit.Quantity(0.179082, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
        ]
        for i in range(3):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            assert charge == water_reference[i][0]
            assert sigma.in_units_of(unit.nanometer) == water_reference[i][
                1
            ].in_units_of(unit.nanometer)
            assert (
                epsilon.in_units_of(unit.kilocalorie_per_mole) == water_reference[i][2]
            )

        # now check methanol
        for i in range(methanol.n_atoms):
            ref_params = methanol.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i + 3)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon

        # now check the water vsite which should be last
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(
            methanol.n_atoms + 3
        )
        assert charge == unit.Quantity(-1.05174, unit.elementary_charge)
        assert sigma == unit.Quantity(1.0, unit.nanometer)
        assert epsilon == unit.Quantity(0.0, unit.kilojoule_per_mole)

        # now check the two methanol sites
        for i, site in enumerate(methanol.extra_sites):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(
                methanol.n_atoms + 4 + i
            )
            assert charge == unit.Quantity(float(site.charge), unit.elementary_charge)
            assert sigma == unit.Quantity(1.0, unit.nanometer)
            assert epsilon == unit.Quantity(0.0, unit.kilojoule_per_mole)


def test_molecule_vsite_and_vsite_water_plugin(methanol, tmpdir, water, rfree_data, vs):
    """
    Make sure a v-site water model is correctly loaded from and offxml
    """
    with tmpdir.as_cwd():

        methanol_copy = methanol.copy(deep=True)

        # run the lj method to make sure the parameters match
        alpha = rfree_data.pop("alpha")
        beta = rfree_data.pop("beta")
        lj = LennardJones612(free_parameters=rfree_data, alpha=alpha, beta=beta)
        # get new Rfree data
        lj.run(methanol_copy)

        # remake rfree
        rfree_data["alpha"] = alpha
        rfree_data["beta"] = beta

        _combine_molecules_offxml(
            molecules=[methanol_copy],
            parameters=elements,
            rfree_data=rfree_data,
            filename="combined.offxml",
            water_model="tip4p-fb",
            h_constraints=False,
        )

        combinded_ff = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        # make sure the sites are local and we have no normal sites
        handlers = combinded_ff.registered_parameter_handlers
        assert "LocalCoordinateVirtualSites" in handlers
        assert "VirtualSites" not in handlers

        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(methanol_copy.to_rdkit()),
            ]
        )
        system = combinded_ff.create_openmm_system(topology=mixed_top)
        # make sure we have 3 constraints
        assert system.getNumConstraints() == 3
        # check each constraint
        reference_constraints = [
            [0, 1, unit.Quantity(0.9572, unit=unit.angstroms)],
            [0, 2, unit.Quantity(0.9572, unit=unit.angstroms)],
            [1, 2, unit.Quantity(1.5139006545247014, unit=unit.angstroms)],
        ]
        for i in range(3):
            a, b, constraint = system.getConstraintParameters(i)
            assert a == reference_constraints[i][0]
            assert b == reference_constraints[i][1]
            assert constraint == reference_constraints[i][2].in_units_of(
                unit.nanometers
            )
        # now loop over the forces and make sure water was correctly parameterised
        forces = dict((force.__class__.__name__, force) for force in system.getForces())
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]
        # first check water has the correct parameters
        water_reference = [
            [
                unit.Quantity(0.0, unit.elementary_charge),
                unit.Quantity(3.1655, unit=unit.angstroms),
                unit.Quantity(0.179082, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
            [
                unit.Quantity(0.52587, unit.elementary_charge),
                unit.Quantity(1, unit=unit.angstroms),
                unit.Quantity(0, unit=unit.kilocalorie_per_mole),
            ],
        ]
        for i in range(3):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            assert charge == water_reference[i][0]
            assert sigma.in_units_of(unit.nanometer) == water_reference[i][
                1
            ].in_units_of(unit.nanometer)
            assert (
                epsilon.in_units_of(unit.kilocalorie_per_mole) == water_reference[i][2]
            )

        # now check methanol
        for i in range(methanol_copy.n_atoms):
            ref_params = methanol_copy.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i + 3)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon

        # now check the water vsite which should be last
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(
            methanol.n_atoms + 3
        )
        assert charge == unit.Quantity(-1.05174, unit.elementary_charge)
        assert sigma == unit.Quantity(1.0, unit.nanometer)
        assert epsilon == unit.Quantity(0.0, unit.kilojoule_per_mole)

        # now check the two methanol sites
        for i, site in enumerate(methanol_copy.extra_sites):
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(
                methanol_copy.n_atoms + 4 + i
            )
            assert charge == unit.Quantity(float(site.charge), unit.elementary_charge)
            assert sigma == unit.Quantity(1.0, unit.nanometer)
            assert epsilon == unit.Quantity(0.0, unit.kilojoule_per_mole)


def test_combine_molecules_offxml_plugin_deepdiff(tmpdir, coumarin, rfree_data):
    """Make sure that systems made from molecules using the xml method match offxmls with plugins"""

    # we need to recalculate the Nonbonded terms using the fake Rfree
    coumarin_copy = coumarin.copy(deep=True)
    # apply symmetry to make sure systems match
    MBISCharges.apply_symmetrisation(coumarin_copy)
    with tmpdir.as_cwd():
        # make the offxml using the plugin interface
        _combine_molecules_offxml(
            molecules=[coumarin_copy],
            parameters=elements,
            rfree_data=rfree_data,
            filename="openff.offxml",
            water_model="tip3p",
            h_constraints=False,
        )
        offxml = ForceField(
            "openff.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        # check the plugin is being used
        vdw = offxml.get_parameter_handler("QUBEKitvdWTS")
        # make sure we have the parameterize tags
        assert len(vdw._cosmetic_attribs) == 1
        assert len(vdw.parameters) == 28

        alpha = rfree_data.pop("alpha")
        beta = rfree_data.pop("beta")
        lj = LennardJones612(free_parameters=rfree_data, alpha=alpha, beta=beta)
        # get new Rfree data
        lj.run(coumarin_copy)
        coumarin_copy.write_parameters("coumarin.xml")
        coumarin_ref_system = xmltodict.parse(
            XmlSerializer.serialize(
                app.ForceField("coumarin.xml").createSystem(
                    topology=coumarin_copy.to_openmm_topology(),
                    nonbondedCutoff=9 * unit.angstroms,
                    removeCMMotion=False,
                )
            )
        )
        coumarin_off_system = xmltodict.parse(
            XmlSerializer.serialize(
                offxml.create_openmm_system(
                    Molecule.from_rdkit(coumarin_copy.to_rdkit()).to_topology()
                )
            )
        )

        coumarin_diff = DeepDiff(
            coumarin_ref_system,
            coumarin_off_system,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        assert len(coumarin_diff) == 1
        for item in coumarin_diff["iterable_item_added"].values():
            assert item["@k"] == "0"


def test_get_water_fail():
    """Make sure an error is rasied if we requested a non-supported water model"""

    with pytest.raises(NotImplementedError):
        _add_water_model(
            ForceField("openff_unconstrained-2.0.0.offxml"),
            water_model="tip4p",
        )


def test_offxml_constraints(acetone, water, openff, tmpdir, rfree_data):
    """Make sure systems made via combined offxml have h-bond constraints when requested."""

    with tmpdir.as_cwd():

        openff.run(acetone)
        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(acetone.to_rdkit()),
            ]
        )

        _combine_molecules_offxml(
            molecules=[acetone],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
            h_constraints=True,
            water_model="tip3p",
        )

        combinded_ff = ForceField("combined.offxml")
        system = combinded_ff.create_openmm_system(topology=mixed_top)
        # make sure we have 3 constraints
        assert system.getNumConstraints() == 9
        # check each water constraint
        reference_constraints = [
            [0, 1, unit.Quantity(0.9572, unit=unit.angstroms)],
            [0, 2, unit.Quantity(0.9572, unit=unit.angstroms)],
            [1, 2, unit.Quantity(1.5139006545247014, unit=unit.angstroms)],
        ]
        for i in range(3):
            a, b, constraint = system.getConstraintParameters(i)
            assert a == reference_constraints[i][0]
            assert b == reference_constraints[i][1]
            assert constraint == reference_constraints[i][2].in_units_of(
                unit.nanometers
            )
        # now check the acetone h-bond constraints
        for i in range(6):
            a, b, constraint = system.getConstraintParameters(i + 3)
            assert (
                unit.Quantity(acetone.BondForce[(a - 3, b - 3)].length, unit.nanometer)
                == constraint
            )


def test_offxml_charges_correctly_applied(tmpdir, openff, rfree_data):
    """
    Make sure that charges are correctly applied when we have similar molecules
    """

    mol1 = Ligand.from_smiles("CSC", "mol1")
    mol2 = Ligand.from_smiles("CS(=O)C", "mol2")

    with tmpdir.as_cwd():
        openff.run(mol1)
        openff.run(mol2)

        # make a combined FF with mol2 first as mol2 is a substructure of mol1 it could overwrite the charges
        _combine_molecules_offxml(
            molecules=[mol2, mol1],
            parameters=[],
            rfree_data=rfree_data,
            filename="charge_test.offxml",
            h_constraints=False,
        )

        ff = ForceField(
            "charge_test.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        off_mol = Molecule.from_rdkit(mol2.to_rdkit())
        system = ff.create_openmm_system(off_mol.to_topology())
        # if the system has been made the post process charge check has worked, but confirm it by testing here
        for force in system.getForces():
            if type(force) == openmm.NonbondedForce:
                break

        for i in range(force.getNumParticles()):
            charge, sigma, epsilon = force.getParticleParameters(i)
            qube_parameter = mol2.NonbondedForce[(i,)]
            # we need to remove the decimal for the comparison
            assert charge == float(qube_parameter.charge) * unit.elementary_charge
            assert sigma == qube_parameter.sigma * unit.nanometer
            assert epsilon == qube_parameter.epsilon * unit.kilojoule_per_mole


@pytest.mark.parametrize(
    "use_local_sites", [pytest.param(True, id="True"), pytest.param(False, id="False")]
)
def test_offxml_water_sites(water, use_local_sites):
    """
    Make sure that the tip4p-fb water model correctly places the vsite relative to the other atoms
    """

    offxml = ForceField(load_plugins=True)
    # Make sure we are using our plugin
    _add_water_model(
        force_field=offxml, water_model="tip4p-fb", use_local_sites=use_local_sites
    )
    off_water = Molecule.from_rdkit(water.to_rdkit())
    system, off_top = offxml.create_openmm_system(
        off_water.to_topology(), return_topology=True
    )

    # create the openmm topology and manually add the vsite as the toolkit drops this info
    openmm_top = off_top.to_openmm()
    for chain in openmm_top.chains():
        if len(chain) == 1:
            residue = chain._residues[0]
            break

    openmm_top.addAtom("EP1", app.Element.getByMass(0), residue)
    positions = off_water.conformers[0].value_in_unit(unit.angstrom)
    padded_pos = np.vstack([positions, [0, 0, 0]])
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.5 * unit.femtosecond
    )
    context = openmm.Context(system, integrator)
    context.setPositions(padded_pos * unit.angstrom)
    # apply the constraints to compute the water and vsite positions
    context.applyConstraints(0)
    state = context.getState(getPositions=True)
    new_positions = state.getPositions().value_in_unit(unit.angstroms)
    # make sure the site is at the expected position
    assert np.allclose(
        new_positions[-1],
        [0.044661788269877434, 0.28491150587797165, 0.043212613090872765],
    )


def test_offxml_combine_no_polar_lj(tmpdir, methanol, rfree_data, vs):
    """
    Make sure the polar H have no LJ terms when we request them to be transferred using the flag
    """

    with tmpdir.as_cwd():

        # run the lj method to make sure the parameters match
        alpha = rfree_data.pop("alpha")
        beta = rfree_data.pop("beta")
        lj = LennardJones612(
            free_parameters=rfree_data, alpha=alpha, beta=beta, lj_on_polar_h=False
        )
        # get new Rfree data
        lj.run(methanol)

        # remake rfree
        rfree_data["alpha"] = alpha
        rfree_data["beta"] = beta

        _combine_molecules_offxml(
            molecules=[methanol],
            parameters=elements,
            rfree_data=rfree_data,
            filename="no_polar_h.offxml",
            water_model="tip4p-fb",
            h_constraints=True,
            lj_on_polar_h=False,
        )
        ff = ForceField(
            "no_polar_h.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )

        vdw = ff.get_parameter_handler("QUBEKitvdWTS")
        assert vdw.lj_on_polar_h == "False"
        assert "parameterize" in vdw._cosmetic_attribs
        assert "xfree" not in getattr(vdw, "_parameterize").split(",")

        # make the OpenMM system and make sure this setting is respected
        off_mol = Molecule.from_rdkit(methanol.to_rdkit())
        openmm_system = ff.create_openmm_system(off_mol.to_topology())

        forces = dict(
            (force.__class__.__name__, force) for force in openmm_system.getForces()
        )
        nonbonded_force: openmm.NonbondedForce = forces["NonbondedForce"]
        # now check methanol
        for i in range(methanol.n_atoms):
            ref_params = methanol.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon
