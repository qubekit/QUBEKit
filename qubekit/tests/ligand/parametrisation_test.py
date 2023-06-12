import openmm
import pytest
import xmltodict
from deepdiff import DeepDiff
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openmm import XmlSerializer, app, unit
from parmed.openmm import energy_decomposition_system, load_topology
from pydantic.error_wrappers import ValidationError

import qubekit
from qubekit.molecules import Ligand
from qubekit.parametrisation import XML, OpenFF
from qubekit.utils.file_handling import get_data


def test_xml_version(openff, acetone, tmpdir):
    """
    Make sure the qubekit version used to make an xml is included in the file.
    """
    with tmpdir.as_cwd():
        openff.run(acetone)
        ff = acetone._build_forcefield()
        root = ff.find("QUBEKit")
        assert qubekit.__version__ == root.get("Version")


@pytest.mark.parametrize(
    "parameter_engine",
    [
        pytest.param("antechamber", id="Antechamber"),
        pytest.param("openff", id="Openff"),
    ],
)
def test_parameter_engines(tmpdir, parameter_engine, openff, antechamber):
    """
    Make sure we can parametrise a molecule using antechamber
    """
    if parameter_engine == "openff":
        engine = openff
    else:
        engine = antechamber

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("acetone.sdf"))

        # make sure we have no starting parameters
        assert mol.BondForce.n_parameters == 0
        assert mol.AngleForce.n_parameters == 0
        assert mol.TorsionForce.n_parameters == 0
        assert mol.ImproperTorsionForce.n_parameters == 0
        assert mol.NonbondedForce.n_parameters == 0

        engine.run(mol)
        # make sure the parameters have been set
        assert mol.BondForce.n_parameters != 0
        assert mol.AngleForce.n_parameters != 0
        assert mol.TorsionForce.n_parameters != 0
        assert mol.ImproperTorsionForce.n_parameters != 0
        assert mol.NonbondedForce.n_parameters != 0


def test_symm_params_no_angles(tmpdir, openff):
    """Make sure we can still apply bonded parameter symmetrisation for linear molecules with no angles."""

    with tmpdir.as_cwd():
        mol = Ligand.from_smiles("Br", "BrH")
        openff.run(mol)
        mol.symmetrise_bonded_parameters()


def test_symm_nonbonded_params(tmpdir, openff):
    with tmpdir.as_cwd():
        mol = Ligand.from_smiles("C", "methane")
        openff.run(mol)
        mol.symmetrise_nonbonded_parameters()
        # make sure all hydrogens have the same parameters
        p = mol.NonbondedForce[(1,)]
        for i in range(3):
            other_p = mol.NonbondedForce[(2 + i,)]
            assert other_p.charge == p.charge
            assert other_p.epsilon == p.epsilon
            assert other_p.sigma == p.sigma


def test_openff_skeleton(tmpdir, openff):
    """
    Make sure the skeleton method in openff works when we have missing coverage in the openff forcefield.
    This will add generic parameters to the forcefield which should match any missing terms, no charges are generated this way.
    """
    with tmpdir.as_cwd():
        # load a molecule with boron
        mol = Ligand.from_file(get_data("132-Benzodioxaborole.pdb"))
        openff.run(molecule=mol)

        # no charges should be generated
        for i in range(mol.n_atoms):
            assert mol.NonbondedForce[(0,)].charge == 0
        # make sure dummy torsions are kept
        boron_torsion = mol.TorsionForce[(1, 0, 8, 7)]
        assert boron_torsion.k1 == 0
        assert boron_torsion.k2 == 0
        assert boron_torsion.k3 == 0
        assert boron_torsion.k4 == 0


@pytest.mark.parametrize(
    "method",
    [
        pytest.param("antechamber", id="antechamber"),
        pytest.param("openff", id="Openff"),
    ],
)
def test_parameter_round_trip(method, tmpdir, xml, openff, antechamber):
    """
    Check we can parametrise a molecule then write out the same parameters.
    """

    if method == "openff":
        param_method = openff
    else:
        param_method = antechamber

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("acetone.sdf"))
        param_mol = param_method.run(mol)
        with open("serialised_acetone.xml") as old:
            with open("orig.xml", "w") as new:
                new.write(old.read())
        # write out params
        mol.write_parameters(file_name="test.xml")

        # make a second mol
        mol2 = Ligand.from_file(get_data("acetone.sdf"))
        param_mol2 = xml.run(molecule=mol2, input_files=["test.xml"])

        for bond in mol.bonds:
            bond_tuple = (bond.atom1_index, bond.atom2_index)
            bond1 = param_mol.BondForce[bond_tuple]
            bond2 = param_mol2.BondForce[bond_tuple]
            assert bond1.k == pytest.approx(bond2.k)
            assert bond1.length == pytest.approx(bond2.length)

        for angle in mol.angles:
            angle1 = param_mol.AngleForce[angle]
            angle2 = param_mol2.AngleForce[angle]
            assert angle1.k == pytest.approx(angle2.k)
            assert angle2.angle == pytest.approx(angle2.angle)

        for atom in range(mol.n_atoms):
            atom1 = param_mol.NonbondedForce[(atom,)]
            atom2 = param_mol2.NonbondedForce[(atom,)]
            assert atom1.charge == pytest.approx(atom2.charge)
            assert atom1.sigma == pytest.approx(atom2.sigma)
            assert atom1.epsilon == pytest.approx(atom2.epsilon)

        # loop over the round trip mol as we lose some parameters which are 0
        for dihedral in param_mol2.TorsionForce.parameters:
            other_dih = param_mol.TorsionForce[dihedral.atoms]
            for key in dihedral.__fields__:
                # openmm will not load any torsions which have k=0, this causes differences between antechamber and
                # qubekit when the phase is not as expected, this does not change the energy however as k=0
                if (
                    key not in ["atoms", "attributes", "parameter_eval"]
                    and "phase" not in key
                ):
                    assert getattr(dihedral, key) == pytest.approx(
                        getattr(other_dih, key)
                    )


def test_improper_round_trip_openff(tmpdir, openff):
    """
    Make sure that improper torsions are correctly round tripped.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("benzene.sdf"))
        # assign new parameters
        openff.run(molecule=mol)
        # make sure we have some 18 impropers
        assert mol.ImproperTorsionForce.n_parameters == 18
        # now write out the parameters
        mol.write_parameters(file_name="test.xml")
        # now load a new molecule
        mol2 = Ligand.from_file(get_data("benzene.sdf"))
        assert mol2.TorsionForce.n_parameters == 0
        XML().run(molecule=mol2, input_files=["test.xml"])
        # make sure we have the same 18 impropers
        assert mol2.ImproperTorsionForce.n_parameters == 18


def test_improper_round_trip_antechamber(tmpdir, antechamber):
    """
    Make sure that improper torsions are correctly round tripped.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("benzene.sdf"))
        # assign new parameters
        antechamber.run(molecule=mol)
        # make sure we have some 6 impropers
        assert mol.ImproperTorsionForce.n_parameters == 6
        # now write out the parameters
        mol.write_parameters(file_name="test.xml")
        # now load a new molecule
        mol2 = Ligand.from_file(get_data("benzene.sdf"))
        assert mol2.TorsionForce.n_parameters == 0
        XML().run(molecule=mol2, input_files=["test.xml"])
        # make sure we have the same 18 impropers
        assert mol2.ImproperTorsionForce.n_parameters == 6


def test_xml_with_sites(tmpdir, xml):
    """
    Make sure that virtual sites are saved into the ligand if they are found in the input file.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("pyridine.pdb"))
        xml.run(
            mol,
            input_files=[
                get_data("pyridine.xml"),
            ],
        )

        assert mol.extra_sites.n_sites == 1
        # make sure that charge we extracted
        sites = mol.extra_sites.get_sites(parent_index=3)
        assert float(sites[0].charge) == -0.180000


def test_xml_sites_roundtrip(tmpdir, xml):
    """
    If we load in an xml with sites make sure we can write it back out.
    """

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("pyridine.pdb"))
        xml.run(
            mol,
            input_files=[
                get_data("pyridine.xml"),
            ],
        )

        mol.write_parameters(file_name="test.xml")

        mol2 = Ligand.from_file(get_data("pyridine.pdb"))
        xml.run(
            mol2,
            input_files=[
                "test.xml",
            ],
        )

        for bond in mol.bonds:
            bond_tuple = (bond.atom1_index, bond.atom2_index)
            bond1 = mol.BondForce[bond_tuple]
            bond2 = mol2.BondForce[bond_tuple]
            assert bond1.k == pytest.approx(bond2.k)
            assert bond1.length == pytest.approx(bond2.length)

        for angle in mol.angles:
            angle1 = mol.AngleForce[angle]
            angle2 = mol2.AngleForce[angle]
            assert angle1.k == pytest.approx(angle2.k)
            assert angle2.angle == pytest.approx(angle2.angle)

        for atom in range(mol.n_atoms):
            atom1 = mol.NonbondedForce[(atom,)]
            atom2 = mol2.NonbondedForce[(atom,)]
            assert atom1.charge == pytest.approx(atom2.charge)
            assert atom1.sigma == pytest.approx(atom2.sigma)
            assert atom1.epsilon == pytest.approx(atom2.epsilon)

        # make sure the virtual site was round tripped
        mol_site = mol.extra_sites.get_sites(parent_index=3)[0]
        mol2_site = mol2.extra_sites.get_sites(parent_index=3)[0]
        assert mol_site.charge == mol2_site.charge
        assert mol_site.parent_index == mol2_site.parent_index
        assert mol_site.closest_a_index == mol2_site.closest_a_index
        assert mol_site.closest_b_index == mol2_site.closest_b_index
        assert mol_site.p1 == mol2_site.p1
        assert mol_site.p2 == mol2_site.p2
        assert mol_site.p3 == mol2_site.p3

        for dihedral in mol.TorsionForce.parameters:
            other_dih = mol2.TorsionForce[dihedral.atoms]
            for key in dihedral.__fields__:
                if key not in ["atoms", "attributes", "parameter_eval"]:
                    assert getattr(dihedral, key) == pytest.approx(
                        getattr(other_dih, key)
                    )


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param("bace0.sdf", id="bace"),
        pytest.param("benzene.sdf", id="benzene"),
        pytest.param("acetone.sdf", id="acetone"),
        pytest.param("pyridine.sdf", id="pyridine"),
    ],
)
@pytest.mark.parametrize(
    "method",
    [
        pytest.param("openff", id="openff"),
        pytest.param("antechamber", id="antechamber"),
    ],
)
def test_round_trip_energy(tmpdir, molecule, method, openff, antechamber):
    """
    Make sure that no terms are missing when storing parameters from source by comparing energies.

    Note we relax the comparison to abs=2e-3 due to differences in nonbonded cutoffs, phase rounding and the ordering
    improper torsions are applied.
    """
    if method == "openff":
        engine = openff
    else:
        engine = antechamber

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data(molecule))
        # parametrise the system
        engine.run(mol)
        # this will make a serialised system in the folder so get the reference energy
        ref_system = XmlSerializer.deserializeSystem(
            open(f"serialised_{mol.name}.xml").read()
        )
        parm_top = load_topology(
            mol.to_openmm_topology(), system=ref_system, xyz=mol.openmm_coordinates()
        )
        ref_energy = energy_decomposition_system(
            parm_top, ref_system, platform="Reference"
        )
        # now we need to build the system from our stored parameters
        mol.write_parameters(file_name="test.xml")
        ff = app.ForceField("test.xml")
        qube_system = ff.createSystem(mol.to_openmm_topology())
        with open("qube.xml", "w") as xml_out:
            xml_out.write(XmlSerializer.serialize(qube_system))
        qube_struc = load_topology(
            mol.to_openmm_topology(), system=qube_system, xyz=mol.openmm_coordinates()
        )
        qube_energy = energy_decomposition_system(
            qube_struc, qube_system, platform="Reference"
        )
        # compare the decomposed energies of the groups
        for force_group, energy in ref_energy:
            for qube_force, qube_e in qube_energy:
                if force_group == qube_force:
                    assert energy == pytest.approx(qube_e, abs=2e-3)


def test_openff_impropers(tmpdir, openff):
    """
    Make sure we correctly capture smirnoff style trefoil improper torsions.
    see here <https://open-forcefield-toolkit.readthedocs.io/en/0.5.0/smirnoff.html#impropertorsions>
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("benzene.sdf"))
        openff.run(molecule=mol)
        assert mol.TorsionForce.ordering == "smirnoff"
        # 6 impropers applied as trefoil = 18
        assert mol.ImproperTorsionForce.n_parameters == 18


def test_amber_improper(tmpdir, antechamber):
    """
    Make sure we correctly capture the normal amber style improper torsions.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("benzene.sdf"))
        antechamber.run(molecule=mol)
        assert mol.TorsionForce.ordering == "amber"
        assert mol.ImproperTorsionForce.n_parameters == 6
        assert mol.TorsionForce.n_parameters == 24
        assert mol.RBTorsionForce.n_parameters == 0
        assert mol.ImproperRBTorsionForce.n_parameters == 0


def test_param_storage(tmpdir):
    """
    Make sure the new parameter storage can be accessed and raises an error when creating parameters
    with incomplete data.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("chloromethane.pdb"))
        OpenFF().run(mol)
        with pytest.raises(ValidationError):
            # Try to only set one param at once (create requires all at once)
            mol.NonbondedForce.create_parameter(atoms=(0,), charge=0.1)
        mol.NonbondedForce.create_parameter(
            atoms=(0,), charge=0.1, epsilon=0.2, sigma=0.3
        )

        assert float(mol.NonbondedForce[(0,)].charge) == 0.1
        assert mol.NonbondedForce[(0,)].epsilon == 0.2
        assert mol.NonbondedForce[(0,)].sigma == 0.3

        mol.NonbondedForce[(0,)].charge = 5
        assert float(mol.NonbondedForce[(0,)].charge) == 5

        assert mol.BondForce[(0, 1)].k == mol.BondForce[(1, 0)].k


@pytest.mark.parametrize(
    "force_group, key, ff_group, terms",
    [
        pytest.param(
            "BondForce",
            "Bond",
            "HarmonicBondForce",
            [(0, 2), (2, 6), (4, 16), (6, 18)],
            id="HarmonicBondForce",
        ),
        pytest.param(
            "AngleForce",
            "Angle",
            "HarmonicAngleForce",
            [(2, 0, 3), (0, 3, 7), (1, 4, 8), (4, 8, 20)],
            id="HarmonicAngleForce",
        ),
        pytest.param(
            "TorsionForce",
            "Proper",
            "PeriodicTorsionForce",
            [(15, 3, 7, 10), (1, 4, 8, 20), (19, 7, 10, 11)],
            id="PeriodicTorsionForce",
        ),
    ],
)
def test_parameter_tags(tmpdir, force_group, ff_group, key, terms):
    """
    Make sure that the parameter tagger tags correct terms.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("biphenyl.sdf"))
        OpenFF().run(molecule=mol)
        # set the parameter tags
        for term in terms:
            f_group = getattr(mol, force_group)
            parameter = f_group[term]
            parameter.attributes = {"test tag"}
        # make the force field
        ff = mol._build_forcefield()
        classes = [
            [f"{mol.atoms[i].atomic_symbol}{mol.atoms[i].atom_index}" for i in term]
            for term in terms
        ]
        term_length = len(terms[0])
        # now search through and make sure the force groups were tagged
        for group in ff.iter(tag=ff_group):
            for ff_term in group.iter(tag=key):
                ff_class = [ff_term.get(f"class{i}") for i in range(1, 1 + term_length)]
                if ff_class in classes:
                    assert ff_term.get("parametrize") == "test tag"
                else:
                    assert ff_term.get("parametrize", None) is None


def test_read_rb_terms(tmpdir):
    """
    Make sure we can parameterise a molecules using an xml file with RB torsion terms.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("cyclohexane.sdf"))
        XML().run(molecule=mol, input_files=[get_data("cyclohexane.xml")])
        # there should be 6 RB terms
        assert mol.RBTorsionForce.n_parameters == 6
        # and no periodic terms
        assert mol.TorsionForce.n_parameters == 0


def test_rb_energy_round_trip(tmpdir):
    """
    Make sure that no parameters are lost when reading in  RBterms.
    """
    with tmpdir.as_cwd():
        # load the molecule and parameterise
        mol = Ligand.from_file(file_name=get_data("cyclohexane.sdf"))
        XML().run(molecule=mol, input_files=[get_data("cyclohexane.xml")])
        # load the serialised system we extract the parameters from as our reference
        ref_system = XmlSerializer.deserializeSystem(
            open(f"serialised_{mol.name}.xml").read()
        )
        parm_top = load_topology(
            mol.to_openmm_topology(), system=ref_system, xyz=mol.openmm_coordinates()
        )
        ref_energy = energy_decomposition_system(
            parm_top, ref_system, platform="Reference"
        )
        # now we need to build the system from our stored parameters
        mol.write_parameters(file_name="test.xml")
        ff = app.ForceField("test.xml")
        qube_system = ff.createSystem(mol.to_openmm_topology())
        with open("qube.xml", "w") as xml_out:
            xml_out.write(XmlSerializer.serialize(qube_system))
        qube_struc = load_topology(
            mol.to_openmm_topology(), system=qube_system, xyz=mol.openmm_coordinates()
        )
        qube_energy = energy_decomposition_system(
            qube_struc, qube_system, platform="Reference"
        )
        # compare the decomposed energies of the groups
        for force_group, energy in ref_energy:
            for qube_force, qube_e in qube_energy:
                if force_group == qube_force:
                    assert energy == pytest.approx(qube_e, abs=2e-3)


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param("bace0.sdf", id="bace"),
        pytest.param("benzene.sdf", id="benzene"),
        pytest.param("acetone.sdf", id="acetone"),
        pytest.param("pyridine.sdf", id="pyridine"),
        pytest.param("methane.sdf", id="methane"),
    ],
)
def test_offxml_round_trip(tmpdir, openff, molecule):
    """
    Test round tripping offxml parameters through qubekit
    """

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data(molecule))
        offmol = Molecule.from_file(get_data(molecule))
        openff.run(mol)
        mol.to_offxml("test.offxml", h_constraints=False)
        # build another openmm system and serialise to compare with deepdiff
        offxml = ForceField("test.offxml")
        assert offxml.author == f"QUBEKit_version_{qubekit.__version__}"
        qubekit_system = offxml.create_openmm_system(topology=offmol.to_topology())
        qubekit_xml = xmltodict.parse(openmm.XmlSerializer.serialize(qubekit_system))
        with open("qubekit_xml", "w") as output:
            output.write(openmm.XmlSerializer.serialize(qubekit_system))
        openff_system = xmltodict.parse(open(f"serialised_{mol.name}.xml").read())

        offxml_diff = DeepDiff(
            qubekit_xml,
            openff_system,
            ignore_order=True,
            significant_digits=6,
        )
        # the only difference should be in torsions with a 0 barrier height which are excluded from an offxml
        if "iterable_item_removed" in offxml_diff:
            for item in offxml_diff["iterable_item_removed"].values():
                assert item["@k"] == "0"

        # load both systems and compute the energy
        qubekit_top = load_topology(
            mol.to_openmm_topology(),
            system=qubekit_system,
            xyz=mol.openmm_coordinates(),
        )
        qubekit_energy = energy_decomposition_system(
            qubekit_top, qubekit_system, platform="Reference"
        )

        ref_system = XmlSerializer.deserializeSystem(
            open(f"serialised_{mol.name}.xml").read()
        )
        parm_top = load_topology(
            mol.to_openmm_topology(), system=ref_system, xyz=mol.openmm_coordinates()
        )
        ref_energy = energy_decomposition_system(
            parm_top, ref_system, platform="Reference"
        )
        # compare the decomposed energies of the groups
        for force_group, energy in ref_energy:
            for qube_force, qube_e in qubekit_energy:
                if force_group == qube_force:
                    assert energy == pytest.approx(qube_e, abs=2e-3)


def test_no_ub_terms_default(methanol):
    """Make sure the default xml has no UB handler when there are no UB terms."""

    # make sure methanol has no us terms
    assert methanol.UreyBradleyForce.n_parameters == 0
    ff = methanol._build_forcefield().getroot()
    force = ff.find("AmoebaUreyBradleyForce")
    assert force is None

    # add them in and make sure they are present
    for angle in methanol.angles:
        methanol.UreyBradleyForce.create_parameter(angle, k=1, d=2)
    ff = methanol._build_forcefield().getroot()
    force = ff.find("AmoebaUreyBradleyForce")
    assert force.tag == "AmoebaUreyBradleyForce"


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param("bace0.sdf", id="bace"),
        pytest.param("acetone.sdf", id="acetone"),
        pytest.param("pyridine.sdf", id="pyridine"),
        pytest.param("methane.sdf", id="methane"),
    ],
)
def test_offxml_round_trip_ub_terms(tmpdir, openff, molecule):
    """
    Test round tripping offxml parameters through qubekit with urey-bradley terms
    """

    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data(molecule))
        offmol = Molecule.from_file(get_data(molecule))
        openff.run(mol)
        # Add urey-bradley terms
        for angle in mol.angles:
            mol.UreyBradleyForce.create_parameter(angle, k=1, d=2)

        mol.to_offxml("test.offxml", h_constraints=False)
        # build another openmm system and serialise to compare with deepdiff
        offxml = ForceField("test.offxml")
        assert offxml.author == f"QUBEKit_version_{qubekit.__version__}"
        qubekit_offxml_system = offxml.create_openmm_system(
            topology=offmol.to_topology()
        )
        qubekit_offxml_xml = xmltodict.parse(
            openmm.XmlSerializer.serialize(qubekit_offxml_system)
        )
        with open("qubekit_offxml_system.xml", "w") as output:
            output.write(openmm.XmlSerializer.serialize(qubekit_offxml_system))

        # now write the normal xml
        mol.write_parameters("test.xml")
        xml_ff = app.ForceField("test.xml")
        qubekit_xml_system = xml_ff.createSystem(
            mol.to_openmm_topology(),
            nonbondedCutoff=0.9 * unit.nanometers,
            removeCMMotion=False,
        )

        qubekit_xml_xml = xmltodict.parse(
            openmm.XmlSerializer.serialize(qubekit_xml_system)
        )
        with open("qubekit_xml_system.xml", "w") as output:
            output.write(openmm.XmlSerializer.serialize(qubekit_xml_system))

        offxml_diff = DeepDiff(
            qubekit_offxml_xml,
            qubekit_xml_xml,
            ignore_order=True,
            significant_digits=6,
            exclude_regex_paths="mass",
        )
        if molecule == "methane.sdf":
            assert len(offxml_diff) == 0
        else:
            assert len(offxml_diff) == 1
            # the only difference should be in torsions with a 0 barrier height which are excluded from an offxml
            if "iterable_removed_added" in offxml_diff:
                for item in offxml_diff["iterable_removed_added"].values():
                    assert item["@k"] == "0"

        # load both systems and compute the energy
        qubekit_xml_top = load_topology(
            mol.to_openmm_topology(),
            system=qubekit_xml_system,
            xyz=mol.openmm_coordinates(),
        )
        qubekit_energy = energy_decomposition_system(
            qubekit_xml_top, qubekit_xml_system, platform="Reference"
        )

        qubekit_offxml_top = load_topology(
            mol.to_openmm_topology(),
            system=qubekit_offxml_system,
            xyz=mol.openmm_coordinates(),
        )
        offxml_energy = energy_decomposition_system(
            qubekit_offxml_top, qubekit_offxml_system, platform="Reference"
        )
        # compare the decomposed energies of the groups
        for force_group, energy in offxml_energy:
            for qube_force, qube_e in qubekit_energy:
                if force_group == qube_force:
                    assert energy == pytest.approx(qube_e, abs=2e-3)


def test_vsite_handler_labeling(methanol, tmpdir):
    """Make sure our plugin handler can label molecules so as not to break down stream workflows"""

    with tmpdir.as_cwd():
        # write out a offxml with local coord sites
        methanol.to_offxml("methanol.offxml", h_constraints=False)
        ff = ForceField("methanol.offxml", load_plugins=True)
        vsite_handler = ff.get_parameter_handler("LocalCoordinateVirtualSites")
        assert len(vsite_handler.parameters) == 2
        off_mol = Molecule.from_rdkit(methanol.to_rdkit())
        matches = vsite_handler._find_matches(entity=off_mol.to_topology())
        # they should only be on the oxygen atom
        assert len(matches) == 1
        # there should be a lone pair on the oxygen
        assert len(matches[(1,)]) == 2


@pytest.mark.parametrize(
    "molecule",
    [pytest.param("methanol", id="methanol"), pytest.param("ethanol", id="ethanol")],
)
def test_offxml_sites_energy(xml, tmpdir, methanol, ethanol, molecule):
    """Make sure virtual sites are correctly written to offxml
    and produce the same energy as an xml system with sites"""

    molecules = {"methanol": methanol, "ethanol": ethanol}
    mol = molecules[molecule]

    with tmpdir.as_cwd():
        # make sure the molecule has extra sites
        assert mol.extra_sites.n_sites == 2

        openff_mol = Molecule.from_rdkit(mol.to_rdkit())
        mol.to_offxml("test.offxml", h_constraints=False)
        mol.write_parameters("test.xml")
        ff = ForceField("test.offxml", load_plugins=True)
        # check the site handler has two sites
        sites = ff.get_parameter_handler("LocalCoordinateVirtualSites")
        assert len(sites._parameters) == 2
        off_system, off_top = ff.create_openmm_system(
            openff_mol.to_topology(), return_topology=True
        )
        # check we have two virtual sites in the openff system
        v_sites = []
        for i in range(off_system.getNumParticles()):
            if off_system.isVirtualSite(i):
                v_sites.append(off_system.getVirtualSite(i))
        assert len(v_sites) == 2
        # make sure 2 vsites are in the openff topology
        assert off_top.n_topology_virtual_sites == 2

        with open("offxml_system.xml", "w") as out:
            out.write(openmm.XmlSerializer.serialize(off_system))
        off_system_xml = xmltodict.parse(open("offxml_system.xml").read())
        # make our reference xml system
        xml_ff = app.ForceField("test.xml")
        # we don't have access to the openmm topology with vsites so make our own
        modeller = app.Modeller(
            topology=mol.to_openmm_topology(),
            positions=mol.openmm_coordinates(),
        )
        modeller.addExtraParticles(xml_ff)
        xml_system = xml_ff.createSystem(
            modeller.topology, removeCMMotion=False, nonbondedCutoff=0.9
        )
        with open("xml_system.xml", "w") as out:
            out.write(openmm.XmlSerializer.serialize(xml_system))
        xml_system_xml = xmltodict.parse(open("xml_system.xml").read())

        # get the system deepdiff
        offxml_diff = DeepDiff(
            off_system_xml,
            xml_system_xml,
            ignore_order=True,
            exclude_regex_paths="mass",
            significant_digits=6,
        )

        # the only difference should be in torsions with a 0 barrier height which are excluded from an offxml
        if "iterable_item_removed" in offxml_diff:
            for item in offxml_diff["iterable_item_removed"].values():
                assert item["@k"] == "0"

        # check the energies match
        xml_topology = load_topology(
            modeller.topology,
            system=xml_system,
            xyz=modeller.positions,
        )
        xml_energy = energy_decomposition_system(
            xml_topology, xml_system, platform="Reference"
        )

        off_topology = load_topology(
            modeller.topology, system=off_system, xyz=modeller.positions
        )
        off_energy = energy_decomposition_system(
            off_topology, off_system, platform="Reference"
        )

        # compare the decomposed energies of the groups
        for force_group, energy in xml_energy:
            for off_force, off_e in off_energy:
                if force_group == off_force:
                    assert energy == pytest.approx(off_e, abs=2e-3)


def test_rb_offxml_deepdiff(tmpdir):
    """Make sure  we can correctly write out an offxml for a molecule with RBProper torsions
    and get the same system as xml"""

    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("cyclohexane.sdf"))
        XML().run(molecule=mol, input_files=[get_data("cyclohexane.xml")])
        # there should be 6 RB terms
        assert mol.RBTorsionForce.n_parameters == 6
        # load the xml openmm system
        xml_ff = app.ForceField(get_data("cyclohexane.xml"))
        xml_system = xml_ff.createSystem(
            mol.to_openmm_topology(),
            nonbondedCutoff=9 * unit.angstroms,
            removeCMMotion=False,
        )
        system_xml = xmltodict.parse(openmm.XmlSerializer.serialize(xml_system))

        mol.to_offxml("test.offxml", h_constraints=False)
        offxml = ForceField(
            "test.offxml", load_plugins=True, allow_cosmetic_attributes=True
        )
        offmol = Molecule.from_file(get_data("cyclohexane.sdf"))
        system = offxml.create_openmm_system(offmol.to_topology())
        off_system = xmltodict.parse(openmm.XmlSerializer.serialize(system))

        system_diff = DeepDiff(
            system_xml,
            off_system,
            ignore_order=True,
            exclude_regex_paths="mass",
            significant_digits=6,
        )
        # there should only be one bond which is the wrong way around for some reason!
        assert list(system_diff.keys()) == ["values_changed"]
        assert system_diff["values_changed"][
            "root['System']['Forces']['Force'][0]['Bonds']['Bond'][5]['@p1']"
        ] == {"new_value": "0", "old_value": "5"}

        # check the energies match
        xml_topology = load_topology(
            mol.to_openmm_topology(),
            system=xml_system,
            xyz=mol.openmm_coordinates(),
        )
        xml_energy = energy_decomposition_system(
            xml_topology, xml_system, platform="Reference"
        )

        off_topology = load_topology(
            mol.to_openmm_topology(), system=system, xyz=mol.openmm_coordinates()
        )
        off_energy = energy_decomposition_system(
            off_topology, system, platform="Reference"
        )

        # compare the decomposed energies of the groups
        for force_group, energy in xml_energy:
            for off_force, off_e in off_energy:
                if force_group == off_force:
                    assert energy == pytest.approx(off_e, abs=2e-3)


def test_rb_torsion_dummy_offxxml(coumarin_with_rb, tmpdir):
    """
    Make sure a dummy proper torsion parameter is only added to an
    offxml when we have a mixture of proper and RB torsions.
    """

    with tmpdir.as_cwd():
        coumarin_with_rb.to_offxml(file_name="coumarin.offxml", h_constraints=False)
        ff = ForceField("coumarin.offxml", load_plugins=True)
        # the force field should use a mixture of proper and rb torsions
        proper_torsions = ff.get_parameter_handler("ProperTorsions")
        generic_parameter = proper_torsions["[*:1]~[*:2]~[*:3]~[*:4]"]
        assert generic_parameter.k1 == unit.Quantity(0.0, unit=unit.kilojoule_per_mole)
        # make sure the generic parameter is first
        assert proper_torsions._index_of_parameter(parameter=generic_parameter) == 0
        rb_torsions = ff.get_parameter_handler("ProperRyckhaertBellemans")
        assert len(rb_torsions.parameters) == 6


def test_mixed_rb_proper_offxml(coumarin_with_rb, tmpdir):
    """
    Compare openmm systems made from offxml and normal xml for molecules with mixed proper
    and RB torsions.
    """
    with tmpdir.as_cwd():
        coumarin_with_rb.to_offxml("coumarin.offxml", h_constraints=False)
        coumarin_with_rb.write_parameters("coumarin.xml")
        xml_ff = app.ForceField("coumarin.xml")
        offxml = ForceField("coumarin.offxml", load_plugins=True)
        xml_system = xml_ff.createSystem(
            coumarin_with_rb.to_openmm_topology(),
            nonbondedCutoff=9 * unit.angstroms,
            removeCMMotion=False,
        )

        offmol = Molecule.from_rdkit(coumarin_with_rb.to_rdkit())
        offxml_system = offxml.create_openmm_system(offmol.to_topology())

        # compare the system energies
        xml_topology = load_topology(
            coumarin_with_rb.to_openmm_topology(),
            system=xml_system,
            xyz=coumarin_with_rb.openmm_coordinates(),
        )
        xml_energy = energy_decomposition_system(
            xml_topology, xml_system, platform="Reference"
        )

        off_topology = load_topology(
            coumarin_with_rb.to_openmm_topology(),
            system=offxml_system,
            xyz=coumarin_with_rb.openmm_coordinates(),
        )
        off_energy = energy_decomposition_system(
            off_topology, offxml_system, platform="Reference"
        )

        # compare the decomposed energies of the groups
        for force_group, energy in xml_energy:
            for off_force, off_e in off_energy:
                if force_group == off_force:
                    print(force_group, energy, off_e)
                    assert energy == pytest.approx(off_e, abs=2e-3)


def test_h_constraints_offxml(methanol, tmpdir):
    """Make sure constraints are added to an offxml when requested and they are included in an openmm system."""

    with tmpdir.as_cwd():
        methanol.to_offxml(file_name="methanol.offxml", h_constraints=True)
        methanol_ff = ForceField("methanol.offxml")
        off_methanol = Molecule.from_rdkit(methanol.to_rdkit())
        system = methanol_ff.create_openmm_system(topology=off_methanol.to_topology())
        # make sure all h bonds are constrained
        assert system.getNumConstraints() == 4
        for i in range(4):
            a, b, constraint = system.getConstraintParameters(i)
            # compare the constraint length with the equilibrium bond length
            assert (
                unit.Quantity(methanol.BondForce[(a, b)].length, unit.nanometer)
                == constraint
            )


def test_parameterizable_offxml_no_fragments(biphenyl, openff, tmpdir):
    """
    Check we can write out a parameterizable offxml with tags for the correct torsions
    """

    with tmpdir.as_cwd():
        openff.run(biphenyl)
        file_name = "biphenyl.offxml"
        biphenyl._optimizeable_offxml(file_name=file_name, h_constraints=False)
        ff = ForceField(file_name, load_plugins=True, allow_cosmetic_attributes=True)
        scanned_bond = tuple(sorted(biphenyl.qm_scans[0].central_bond))
        torsions = ff.get_parameter_handler("ProperTorsions")
        for parameter in torsions.parameters:
            if parameter.attribute_is_cosmetic("parameterize"):
                # make sure the parameter matches the rotatable bond
                matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
                for match in matches:
                    assert tuple(sorted(match[1:3])) == scanned_bond


def test_parameterizable_offxml_fragments(biphenyl_fragments, tmpdir):
    """
    Check we can write out a parameterizable offxml with tags for the correct torsions which match both the
    frgament and the parent.
    """

    with tmpdir.as_cwd():
        file_name = "ring_test.offxml"
        # we know the molecule has one fragment
        fragment = biphenyl_fragments.fragments[0]
        biphenyl_fragments._optimizeable_offxml(
            file_name=file_name, h_constraints=False
        )
        ff = ForceField(file_name, load_plugins=True, allow_cosmetic_attributes=True)
        scanned_fragment_bond = tuple(sorted(fragment.qm_scans[0].central_bond))
        torsions = ff.get_parameter_handler("ProperTorsions")
        for parameter in torsions.parameters:
            if parameter.attribute_is_cosmetic("parameterize"):
                # make sure the parameter matches the rotatable bond in the fragment and parent
                fragment_matches = fragment.get_smarts_matches(smirks=parameter.smirks)
                parent_matches = biphenyl_fragments.get_smarts_matches(
                    smirks=parameter.smirks
                )
                for match in fragment_matches:
                    # check it hits the target bond
                    assert tuple(sorted(match[1:3])) == scanned_fragment_bond
                    # get the map index of the match
                    map_match = [fragment.atoms[i].map_index for i in match]
                    # get the parent torsion
                    parent_torsion = [
                        biphenyl_fragments.get_atom_with_map_index(i).atom_index
                        for i in map_match
                    ]
                    # check this torsion was also hit in the parent
                    if (
                        tuple(parent_torsion) not in parent_matches
                        and tuple(reversed(parent_torsion)) not in parent_matches
                    ):
                        raise RuntimeError(
                            f"parent torsion {parent_torsion} not matched by smirks {parent_matches}"
                        )


def test_offxml_clustered_types(biphenyl, openff, tmpdir):
    """
    Test that using a chemper SingleGraph to create the smirks types for a single instance of an interaction (bond, angle)
    from a cluster of similar terms hits all intended targets. This was swapped as using ClusterGraph created too general
    terms which caused duplicate parameter errors.
    """

    with tmpdir.as_cwd():
        openff.run(biphenyl)
        file_name = "biphenyl.offxml"
        biphenyl.to_offxml(file_name=file_name, h_constraints=False)
        ff = ForceField(file_name, load_plugins=True, allow_cosmetic_attributes=True)
        # for each handler ensure that each smirks term hits all atoms of the same interaction type
        bonds = ff.get_parameter_handler("Bonds")
        for parameter, expected_matches in zip(
            bonds.parameters, biphenyl.bond_types.values()
        ):
            matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
            for match in expected_matches:
                assert match in matches

        angles = ff.get_parameter_handler("Angles")
        for parameter, expected_matches in zip(
            angles.parameters, biphenyl.angle_types.values()
        ):
            matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
            for match in expected_matches:
                assert match in matches

        impropers = ff.get_parameter_handler("ImproperTorsions")
        for parameter, expected_matches in zip(
            impropers.parameters, biphenyl.improper_types.values()
        ):
            matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
            for match in expected_matches:
                # improper SMIRKS have the central atom second but internal lists have it first
                # so swap the atoms before checking
                assert (match[1], match[0], *match[2:]) in matches

        dihedrals = ff.get_parameter_handler("ProperTorsions")
        for parameter, expected_matches in zip(
            dihedrals.parameters, biphenyl.dihedral_types.values()
        ):
            matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
            for match in expected_matches:
                assert match in matches or tuple(reversed(match)) in matches

        vdw = ff.get_parameter_handler("vdW")
        # group the atom types
        atom_types = {}
        for atom_idx, atom_cip in biphenyl.atom_types.items():
            atom_types.setdefault(atom_cip, []).append(atom_idx)
        for parameter, expected_matches in zip(vdw.parameters, atom_types.values()):
            matches = biphenyl.get_smarts_matches(smirks=parameter.smirks)
            for match in expected_matches:
                # turn the match into a tuple
                assert (match,) in matches
