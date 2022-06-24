import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString

import pytest
import xmltodict
from deepdiff import DeepDiff
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import XmlSerializer, app, openmm, unit

import qubekit
from qubekit.charges import MBISCharges
from qubekit.cli.combine import (
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
        acetone_ref_system = xmltodict.parse(open("serialised.xml").read())
        openff.run(coumarin)
        coumarin_ref_system = xmltodict.parse(open("serialised.xml").read())

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
        acetone_ref_system = xmltodict.parse(open("serialised.xml").read())
        pyridine = Ligand.from_file(file_name=get_data("pyridine.sdf"))
        xml.run(molecule=pyridine, input_files=[get_data("pyridine.xml")])
        pyridine_ref_system = xmltodict.parse(open("serialised.xml").read())

        combined_xml = _combine_molecules(
            molecules=[acetone, pyridine], parameters=elements, rfree_data=rfree_data
        ).getroot()
        messy = ET.tostring(combined_xml, "utf-8")

        pretty_xml = parseString(messy).toprettyxml(indent="")
        print(pretty_xml)
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
            assert len(vdw_handler.parameters) == 34


def test_combine_sites_offxml(xml, tmpdir, rfree_data):
    """
    Make sure an error is raised if we try and combine with vsites.
    """
    with tmpdir.as_cwd():
        pyridine = Ligand.from_file(file_name=get_data("pyridine.sdf"))
        xml.run(molecule=pyridine, input_files=[get_data("pyridine.xml")])
        with pytest.raises(NotImplementedError):
            _combine_molecules_offxml(
                molecules=[pyridine],
                parameters=elements,
                rfree_data=rfree_data,
                filename="test.offxml",
            )


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
        with pytest.raises(NotImplementedError):
            _combine_molecules_offxml(
                molecules=[mol],
                parameters=elements,
                rfree_data=rfree_data,
                filename="test.offxml",
            )


def test_combine_molecules_deepdiff_offxml(
    acetone, openff, coumarin, tmpdir, rfree_data
):
    """When not optimising anything make sure we can round trip openff parameters"""

    with tmpdir.as_cwd():
        openff.run(acetone)
        acetone_ref_system = xmltodict.parse(open("serialised.xml").read())
        openff.run(coumarin)
        coumarin_ref_system = xmltodict.parse(open("serialised.xml").read())

        _combine_molecules_offxml(
            molecules=[acetone, coumarin],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
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


def test_molecule_and_water_offxml(coumarin, water, tmpdir, rfree_data):
    """Test that an offxml can parameterize a molecule and water mixture."""

    with tmpdir.as_cwd():

        _combine_molecules_offxml(
            molecules=[coumarin],
            parameters=[],
            rfree_data=rfree_data,
            filename="combined.offxml",
            water_model="tip3p",
        )

        combinded_ff = ForceField(
            "combined.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        mixed_top = Topology.from_molecules(
            molecules=[
                Molecule.from_rdkit(water.to_rdkit()),
                Molecule.from_rdkit(coumarin.to_rdkit()),
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
        for i in range(coumarin.n_atoms):
            ref_params = coumarin.NonbondedForce[(i,)]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i + 3)
            assert charge.value_in_unit(unit.elementary_charge) == float(
                ref_params.charge
            )
            assert sigma.value_in_unit(unit.nanometers) == ref_params.sigma
            assert epsilon.value_in_unit(unit.kilojoule_per_mole) == ref_params.epsilon


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
