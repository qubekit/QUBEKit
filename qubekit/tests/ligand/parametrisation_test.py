import pytest
from parmed.openmm import energy_decomposition_system, load_topology
from pydantic.error_wrappers import ValidationError
from simtk.openmm import XmlSerializer, app

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
        with open("serialised.xml") as old:
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
                if key not in ["atoms", "attributes", "parameter_eval"]:
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
        ref_system = XmlSerializer.deserializeSystem(open("serialised.xml").read())
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
        ref_system = XmlSerializer.deserializeSystem(open("serialised.xml").read())
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
