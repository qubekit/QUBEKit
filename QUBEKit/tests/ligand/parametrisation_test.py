import numpy as np
import pytest
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.toolkit.typing.engines.smirnoff import ForceField

from QUBEKit.ligand import Ligand
from QUBEKit.parametrisation import XML, AnteChamber, OpenFF, Parametrisation
from QUBEKit.utils.file_handling import get_data


def test_parameter_prep():
    """Test that the base parameter class preps a molecule to store prameters."""
    mol = Ligand(get_data("acetone.pdb"))
    assert mol.AtomTypes is None
    assert mol.HarmonicBondForce is None
    assert mol.HarmonicAngleForce is None
    assert mol.PeriodicTorsionForce is None
    assert mol.NonbondedForce is None

    # now use the base class to prep the molecule
    Parametrisation(mol)
    assert mol.AtomTypes == {}
    assert len(mol.HarmonicBondForce) == mol.n_bonds
    assert len(mol.HarmonicAngleForce) == mol.n_angles
    assert len(mol.NonbondedForce) == mol.n_atoms


def test_antechamber(tmpdir):
    """
    Make sure we can parametrise a molecule using antechamber
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        AnteChamber(mol)

        # loop over the parameters and make sure they not defaults
        for bond in mol.bonds:
            assert mol.HarmonicBondForce[bond] != [0, 0]
        for angle in mol.angles:
            assert mol.HarmonicAngleForce[angle] != [0, 0]
        assert (
            len(mol.PeriodicTorsionForce) == mol.n_dihedrals + mol.n_improper_torsions
        )
        for i in range(mol.n_atoms):
            assert mol.NonbondedForce[i] != [0, 0, 0]


def test_openff(tmpdir):
    """
    Make sure we can parametrise a molecule using openff.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        OpenFF(mol)

        # loop over the parameters and make sure they not defaults
        for bond in mol.bonds:
            assert mol.HarmonicBondForce[bond] != [0, 0]
        for angle in mol.angles:
            assert mol.HarmonicAngleForce[angle] != [0, 0]
        assert (
            len(mol.PeriodicTorsionForce) == mol.n_dihedrals + mol.n_improper_torsions
        )
        for i in range(mol.n_atoms):
            assert mol.NonbondedForce[i] != [0, 0, 0]


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param(
            Ligand.from_file(file_name=get_data("benzene.sdf")), id="From sdf"
        ),
        pytest.param(Ligand.from_smiles("c1ccccc1", name="benzene"), id="From smiles"),
    ],
)
def test_openff_aromaticity(tmpdir, molecule):
    """
    Make sure that we use the correct aromaticity model to type the openff molecule.
    """

    with tmpdir.as_cwd():
        # check aromatic
        for atom in molecule.rdkit_mol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                assert atom.GetIsAromatic() is True
        # build the off molecule
        off_mol = OFFMolecule.from_rdkit(rdmol=molecule.rdkit_mol)
        # build the ff and type the molecule
        ff = ForceField("openff_unconstrained-1.3.0.offxml")
        labels = ff.label_molecules(topology=off_mol.to_topology())[0]
        # get a ring torsion
        torsion = labels["ProperTorsions"][(0, 1, 2, 3)]
        # make sure it hits an aromatic bond
        assert torsion.id == "t44"
        # now check the improper
        improper = labels["ImproperTorsions"][(0, 1, 2, 7)]
        assert improper.id == "i1"


def test_openff_skeleton(tmpdir):
    """
    Make sure the skeleton method in openff works when we have missing coverage in the openff forcefield.
    This will add generic parameters to the forcefield which should match any missing terms, no charges are generated this way.
    """
    with tmpdir.as_cwd():
        # load a molecule with b
        mol = Ligand(get_data("132-Benzodioxaborole.pdb"))
        OpenFF(mol)

        # no charges should be generated
        for i in range(mol.n_atoms):
            assert mol.NonbondedForce[i][0] == 0


@pytest.mark.parametrize(
    "method",
    [pytest.param(AnteChamber, id="antechamber"), pytest.param(OpenFF, id="Openff")],
)
def test_parameter_round_trip(method, tmpdir):
    """
    Check we can parametrise a molecule then write out the same parameters.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("benzene.pdb"))
        method(mol)
        # write out params
        mol.write_parameters(name="test")

        # make a second mol
        mol2 = Ligand(get_data("benzene.pdb"))
        XML(mol2, "test.xml")

        assert mol.AtomTypes == mol2.AtomTypes
        for bond in mol.HarmonicBondForce.keys():
            assert mol.HarmonicBondForce[bond] == pytest.approx(
                mol2.HarmonicBondForce[bond], abs=1e-6
            )
        for angle in mol.HarmonicAngleForce.keys():
            assert mol.HarmonicAngleForce[angle] == pytest.approx(
                mol2.HarmonicAngleForce[angle], abs=1e-6
            )
        for atom in range(mol.n_atoms):
            assert (
                pytest.approx(mol.NonbondedForce[atom], abs=1e-6)
                == mol2.NonbondedForce[atom]
            )
        for dihedral, terms in mol.PeriodicTorsionForce.items():
            try:
                other_dih = mol2.PeriodicTorsionForce[dihedral]
            except KeyError:
                other_dih = mol2.PeriodicTorsionForce[tuple(reversed(dihedral))]
            assert np.allclose(terms[:4], other_dih[:4])


def test_xml_with_sites(tmpdir):
    """
    Make sure that virtual sites are saved into the ligand if they are found in the input file.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("pyridine.pdb"))
        XML(mol, input_file=get_data("pyridine.xml"))

        assert mol.extra_sites is not None
        assert len(mol.extra_sites) == 1
        # make sure that charge we extracted
        assert mol.extra_sites[0].charge == -0.180000


def test_xml_sites_roundtrip(tmpdir):
    """
    If we load in an xml with sites make sure we can write it back out.
    """

    with tmpdir.as_cwd():
        mol = Ligand(get_data("pyridine.pdb"))
        XML(mol, input_file=get_data("pyridine.xml"))

        mol.write_parameters(name="test")

        mol2 = Ligand(get_data("pyridine.pdb"))
        XML(mol2, input_file="test.xml")

        assert mol.AtomTypes == mol2.AtomTypes
        for bond in mol.HarmonicBondForce.keys():
            assert (
                pytest.approx(mol.HarmonicBondForce[bond], abs=1e-6)
                == mol2.HarmonicBondForce[bond]
            )
        for angle in mol.HarmonicAngleForce.keys():
            assert (
                pytest.approx(mol.HarmonicAngleForce[angle], abs=1e-6)
                == mol2.HarmonicAngleForce[angle]
            )
        for atom in range(mol.n_atoms):
            assert (
                pytest.approx(mol.NonbondedForce[atom], abs=1e-6)
                == mol2.NonbondedForce[atom]
            )
        # make sure the virtual site was round tripped
        mol_site = mol.extra_sites[0]
        mol2_site = mol2.extra_sites[0]
        assert mol_site.charge == mol2_site.charge
        assert mol_site.parent_index == mol2_site.parent_index
        assert mol_site.closest_a_index == mol2_site.closest_a_index
        assert mol_site.closest_b_index == mol2_site.closest_b_index
        assert mol_site.p1 == mol2_site.p1
        assert mol_site.p2 == mol2_site.p2
        assert mol_site.p3 == mol2_site.p3

        for dihedral, terms in mol.PeriodicTorsionForce.items():
            try:
                other_dih = mol2.PeriodicTorsionForce[dihedral]
            except KeyError:
                other_dih = mol2.PeriodicTorsionForce[tuple(reversed(dihedral))]
            assert np.allclose(terms[:4], other_dih[:4])


def test_improper_round_trip(tmpdir):
    """
    Make sure that improper torsions are correctly round tripped.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(get_data("benzene.sdf"))
        # assign new parameters
        OpenFF(mol)
        # make sure we have some 6 impropers
        impropers = []
        for key, value in mol.PeriodicTorsionForce.items():
            if value[-1] == "Improper":
                impropers.append((key, value))
        assert len(impropers) == 6, print(impropers)

        # now write out the parameters
        mol.write_parameters(name="test")
        # now load a new molecule
        mol2 = Ligand.from_file(get_data("benzene.sdf"))
        assert mol2.PeriodicTorsionForce is None
        XML(molecule=mol2, input_file="test.xml")
        # make sure we have the same 6 impropers
        impropers2 = []
        for key, value in mol2.PeriodicTorsionForce.items():
            if value[-1] == "Improper":
                impropers2.append((key, value))
        assert len(impropers2) == 6
        assert impropers == impropers2
