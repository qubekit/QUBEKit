import pytest

import numpy as np
from QUBEKit.ligand import Ligand
from QUBEKit.utils.file_handling import get_data
from QUBEKit.parametrisation import AnteChamber, OpenFF, XML, Parametrisation


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
            assert pytest.approx(
                mol.HarmonicBondForce[bond], mol2.HarmonicBondForce[bond]
            )
        for angle in mol.HarmonicAngleForce.keys():
            assert pytest.approx(
                mol.HarmonicAngleForce[angle], mol2.HarmonicAngleForce[angle]
            )
        for atom in range(mol.n_atoms):
            assert pytest.approx(mol.NonbondedForce[atom], mol2.NonbondedForce[atom])
        for dihedral, terms in mol.PeriodicTorsionForce.items():
            try:
                other_dih = mol2.PeriodicTorsionForce[dihedral]
            except KeyError:
                other_dih = mol2.PeriodicTorsionForce[tuple(reversed(dihedral))]
            assert np.allclose(terms[:4], other_dih[:4])
