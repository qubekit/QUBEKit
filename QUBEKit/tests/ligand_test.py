import numpy as np
import pytest
from rdkit.Chem import rdMolTransforms

from QUBEKit.ligand import Ligand
from QUBEKit.utils.file_handling import get_data
from QUBEKit.utils.helpers import unpickle


def test_bonds():
    """Make sure after reading a molecule file that we have the correct set of bonds.
    This is done by comparing back to the rdkit molecule.
    """
    reference = [(0, 4), (0, 5), (0, 6), (0, 1), (1, 2), (1, 3), (3, 7), (3, 8), (3, 9)]
    mol = Ligand(get_data("acetone.pdb"))
    # make sure all bonds present
    for bond in mol.bonds:
        assert bond in reference
    # make sure the number of bonds is the same
    assert len(reference) == mol.n_bonds


def test_angles():
    """
    Make sure we identify all of the angles in acetone.
    """
    reference = [
        (1, 0, 4),
        (1, 0, 5),
        (1, 0, 6),
        (4, 0, 5),
        (4, 0, 6),
        (5, 0, 6),
        (0, 1, 2),
        (0, 1, 3),
        (2, 1, 3),
        (1, 3, 7),
        (1, 3, 8),
        (1, 3, 9),
        (7, 3, 8),
        (7, 3, 9),
        (8, 3, 9),
    ]
    mol = Ligand(get_data("acetone.pdb"))
    for angle in mol.angles:
        assert angle in reference

    assert len(reference) == mol.n_angles


def test_dihedrals():
    """
    Make sure we can find all of the dihedrals in acetone.
    """
    reference = {
        (0, 1): [
            (4, 0, 1, 2),
            (4, 0, 1, 3),
            (5, 0, 1, 2),
            (5, 0, 1, 3),
            (6, 0, 1, 2),
            (6, 0, 1, 3),
        ],
        (1, 3): [
            (0, 1, 3, 7),
            (0, 1, 3, 8),
            (0, 1, 3, 9),
            (2, 1, 3, 7),
            (2, 1, 3, 8),
            (2, 1, 3, 9),
        ],
    }

    mol = Ligand(get_data("acetone.pdb"))
    for bond, dihedrals in mol.dihedrals.items():
        for torsion in dihedrals:
            assert torsion in reference[bond]
        assert len(dihedrals) == len(reference[bond])

    # now check the total number of dihedrals
    assert mol.n_dihedrals == 12


def test_no_dihedrals():
    """
    Make sure we return None when no dihedrals are found in the molecule.
    """
    mol = Ligand(get_data("water.pdb"))
    assert mol.dihedrals is None
    assert mol.n_dihedrals == 0


def test_no_impropers():
    """
    Make sure we return None when no impropers are found in the molecule.
    """
    mol = Ligand(get_data("water.pdb"))
    assert mol.improper_torsions is None
    assert mol.n_improper_torsions == 0


def test_impropers():
    """
    Make sure all improper torsions can be found in acetone.
    note this is fragile as the definition of improper may change.
    """
    reference = [(1, 0, 2, 3)]
    mol = Ligand(get_data("acetone.pdb"))
    for improper in mol.improper_torsions:
        assert improper in reference

    assert mol.n_improper_torsions == len(reference)


def test_coords():
    """
    Make sure that when reading an input file the coordinates are saved.
    """
    mol = Ligand(get_data("acetone.pdb"))
    assert mol.coords["input"].shape == (mol.n_atoms, 3)
    assert mol.coords["qm"] == []
    assert mol.coords["traj"] == []


def test_bond_lengths():
    """
    Make sure we can measure bond lengths for a given conformer and the distances match those given by rdkit.
    """
    mol = Ligand(get_data("acetone.pdb"))
    bond_lengths = mol.measure_bonds(input_type="input")
    rdkit_mol = mol.rdkit_mol
    for bond, length in bond_lengths.items():
        assert pytest.approx(
            rdMolTransforms.GetBondLength(rdkit_mol.GetConformer(), *bond), length
        )


def test_measure_angles():
    """
    Make sure we can correctly measure all of the angles in the molecule.
    """
    mol = Ligand(get_data("acetone.pdb"))
    angle_values = mol.measure_angles(input_type="input")
    rdkit_mol = mol.rdkit_mol
    for angle, value in angle_values.items():
        assert pytest.approx(
            rdMolTransforms.GetAngleDeg(rdkit_mol.GetConformer(), *angle), value
        )


def test_measure_dihedrals():
    """
    Make sure we can correctly measure all dihedrals in the molecule.
    """
    mol = Ligand(get_data("acetone.pdb"))
    dihedral_values = mol.measure_dihedrals(input_type="input")
    rdkit_mol = mol.rdkit_mol
    for dihedral, value in dihedral_values.items():
        assert pytest.approx(
            rdMolTransforms.GetDihedralDeg(rdkit_mol.GetConformer(), *dihedral), value
        )


def test_measure_no_dihedrals():
    """Make sire None is returned when there are no dihedrals to measure."""
    mol = Ligand(get_data("water.pdb"))
    assert mol.measure_dihedrals(input_type="input") is None


def test_get_atom():
    """
    Make sure we can pull the desired atom.
    """
    mol = Ligand(get_data("acetone.pdb"))
    for atom in mol.atoms:
        q_atom = mol.get_atom_with_name(name=atom.atom_name)
        assert q_atom == atom


def test_get_atom_missing():
    """
    If we can not find an atom with this name, make sure to throw an error.
    """
    mol = Ligand(get_data("acetone.pdb"))
    with pytest.raises(AttributeError):
        mol.get_atom_with_name(name="test")


def test_n_atoms():
    """
    Make sure all atoms are loaded into the system correctly.
    """
    mol = Ligand(get_data("acetone.pdb"))
    assert mol.n_atoms == 10


def test_rotatable_bonds_filtered():
    """
    For acetone while there are dihedrals we do not class these as rotatable as they are methyl, make sure
    this is true.
    """
    mol = Ligand(get_data("acetone.pdb"))
    assert mol.rotatable_bonds is None
    assert mol.n_rotatable_bonds == 0


def test_no_rotatable_bonds():
    """
    If there are no dihedrals in the molecule make sire we return None.
    """
    mol = Ligand(get_data("water.pdb"))
    assert mol.rotatable_bonds is None
    assert mol.n_rotatable_bonds == 0


def test_rotatable_bonds():
    """
    Make sure we can find true rotatable bonds for a molecule.
    """
    mol = Ligand(get_data("biphenyl.pdb"))
    assert mol.rotatable_bonds == [
        (3, 4),
    ]
    assert mol.n_rotatable_bonds == 1


def test_atom_types():
    """
    Make sure we can assign atom types to a molecule based on the CIP rank.
    For acetone we epect the following types
    - all hydrogen
    - methyl carbons
    - carbonyl carbon
    - carbonyl oxygen
    """
    mol = Ligand(get_data("acetone.pdb"))
    atom_types = mol.atom_types
    unique_types = set(list(atom_types.values()))
    assert len(unique_types) == 4


def test_bond_types():
    """
    Make sure based on the given atom types we can create bond types.
    """
    mol = Ligand(get_data("acetone.pdb"))
    bond_types = mol.bond_types
    unique_types = set(list(bond_types.keys()))
    assert len(unique_types) == 3


def test_angle_types():
    """
    Make sure based on the given atom types that we can create angle types.
    """
    mol = Ligand(get_data("acetone.pdb"))
    angle_types = mol.angle_types
    unique_types = set(list(angle_types.keys()))
    assert len(unique_types) == 4


def test_dihedral_types():
    """
    Make sure based on the given atom types we can create dihedral types.
    """
    mol = Ligand(get_data("acetone.pdb"))
    dihedral_types = mol.dihedral_types
    unique_types = set(list(dihedral_types.keys()))
    assert len(unique_types) == 2


def test_improper_types():
    """
    Make sure based on the given atom types we can create improper types.
    """
    mol = Ligand(get_data("acetone.pdb"))
    improper_types = mol.improper_types
    unique_types = set(list(improper_types.keys()))
    assert len(unique_types) == 1


def test_repr():
    """Make sure the ligand repr works."""
    mol = Ligand(get_data("water.pdb"))
    repr(mol)


def test_str():
    """
    Make sure that the ligand str method does not raise an error.
    """
    mol = Ligand(get_data("water.pdb"))
    str(mol)


def test_to_openmm_coords():
    """
    Make sure we can convert the coordinates to openmm style coords
    """
    mol = Ligand(get_data("acetone.pdb"))
    coords = mol.openmm_coordinates(input_type="input")
    assert np.allclose(coords[0] * 10, mol.coords["input"])


def test_to_openmm_coords_multiple():
    """
    Make sure we can convert to openmm style coords for multiple conformers.
    """
    mol = Ligand(get_data("butane.pdb"))
    # fake a set of conformers
    coords = [mol.coords["input"], mol.coords["input"]]
    mol.coords["traj"] = coords
    openmm_coords = mol.openmm_coordinates(input_type="traj")
    for i in range(len(openmm_coords)):
        assert np.allclose(openmm_coords[i] * 10, mol.coords["traj"][i])


def test_pickle_round_trip(tmpdir):
    """
    test dumping a molecule to pickle and loading it back in.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        mol.pickle(state="test")
        mols = unpickle()
        pickle_mol = mols["test"]
        for atom in mol.atoms:
            pickle_atom = pickle_mol.get_atom_with_name(atom.atom_name)
            assert pickle_atom.__dict__ == atom.__dict__
        assert mol.bonds == pickle_mol.bonds
        assert mol.angles == pickle_mol.angles
        assert mol.dihedrals == pickle_mol.dihedrals


def test_double_pickle(tmpdir):
    """
    Make sure we can add multiple pickled objects to the same file.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        mol.pickle(state="input")
        # remove all coords
        mol.coords["input"] = []
        mol.pickle(state="after")

        # now check we have both states
        mols = unpickle()
        assert "after" in mols
        assert "input" in mols


def test_write_xyz_single_conformer(tmpdir):
    """
    Write a single conformer xyz file for the molecule.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        mol.write_xyz(input_type="input", name="acetone")

        # now read in and check the file
        with open("acetone.xyz") as xyz:
            lines = xyz.readlines()
            # atoms plus 2 comment lines
            assert len(lines) == mol.n_atoms + 2
            assert float(lines[0]) == mol.n_atoms
            # now loop over the atoms and make sure they match and the coords
            for i, line in enumerate(lines[2:]):
                atom = mol.atoms[i]
                assert atom.atomic_symbol == line.split()[0]
                assert np.allclose(
                    mol.coords["input"][i], [float(coord) for coord in line.split()[1:]]
                )


def test_write_xyz_multiple_conformer(tmpdir):
    """
    Make sure we can write multiple conformer xyz files for a molecule.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("butane.pdb"))
        # fake a set of conformers
        coords = [mol.coords["input"], mol.coords["input"]]
        mol.coords["traj"] = coords
        mol.write_xyz(input_type="traj", name="butane")

        # now read in the file again
        with open("butane.xyz") as xyz:
            lines = xyz.readlines()
            assert len(lines) == 2 * mol.n_atoms + 4


def test_pdb_round_trip(tmpdir):
    """
    Make sure we can write a molecule to pdb and load it back.
    """
    with tmpdir.as_cwd():
        mol = Ligand(get_data("acetone.pdb"))
        mol.write_pdb(input_type="input", name="test")

        mol2 = Ligand("test.pdb")
        for atom in mol.atoms:
            pickle_atom = mol2.get_atom_with_name(atom.atom_name)
            assert pickle_atom.__dict__ == atom.__dict__
        assert mol.bonds == mol2.bonds
        assert mol.angles == mol2.angles
        assert mol.dihedrals == mol2.dihedrals
