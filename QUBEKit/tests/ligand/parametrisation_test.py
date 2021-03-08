import numpy as np
import pytest

from QUBEKit.ligand import Ligand
from QUBEKit.parametrisation import XML, AnteChamber, OpenFF, Parametrisation
from QUBEKit.utils.datastructures import ParameterTags
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
        mol = Ligand(get_data("acetone.pdb"))
        method(mol)
        # write out params
        mol.write_parameters(name="test")

        # make a second mol
        mol2 = Ligand(get_data("acetone.pdb"))
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


@pytest.mark.parametrize(
    "force_group, key, terms",
    [
        pytest.param(
            "HarmonicBondForce",
            "Bond",
            [(0, 2), (2, 6), (4, 16), (6, 18)],
            id="HarmonicBondForce",
        ),
        pytest.param(
            "HarmonicAngleForce",
            "Angle",
            [(2, 0, 3), (0, 3, 7), (1, 4, 8), (4, 8, 20)],
            id="HarmonicAngleForce",
        ),
        pytest.param(
            "PeriodicTorsionForce",
            "Torsion",
            [(15, 3, 7, 10), (1, 4, 8, 20), (19, 7, 10, 11)],
            id="PeriodicTorsionForce",
        ),
    ],
)
def test_parameter_tags(tmpdir, force_group, key, terms):
    """
    Make sure that the parameter tagger tags correct terms.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("biphenyl.sdf"))
        OpenFF(mol)
        # set the parameter tags
        tags = ParameterTags()
        setattr(tags, force_group + "_groups", terms)
        setattr(tags, force_group + "_tags", {"tags": "test tag"})
        # make the force field
        ff = mol.create_forcefield(parameter_tags=tags)
        classes = [
            [f"{mol.atoms[i].atomic_symbol}{mol.atoms[i].atom_index}" for i in term]
            for term in terms
        ]
        term_length = len(terms[0])
        # now search through and make sure the force groups were tagged
        for group in ff.iter(tag=force_group):
            for ff_term in group.iter(tag=key):
                ff_class = [ff_term.get(f"class{i}") for i in range(1, 1 + term_length)]
                if ff_class in classes:
                    assert ff_term.get("tags") == "test tag"
                else:
                    assert ff_term.get("tags", None) is None
