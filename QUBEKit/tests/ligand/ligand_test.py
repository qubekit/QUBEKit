import numpy as np
import pytest
from openff.toolkit.topology import Molecule as OFFMolecule
from rdkit.Chem import rdMolTransforms

from QUBEKit.ligand import Ligand
from QUBEKit.utils.exceptions import FileTypeError
from QUBEKit.utils.file_handling import get_data
from QUBEKit.utils.helpers import unpickle


@pytest.fixture
def acetone():
    """
    Make a ligand class from the acetone pdb.
    """
    return Ligand(get_data("acetone.pdb"))


def test_has_unique_names(acetone):
    """
    Make sure the method can correctly identify if a molecule has unique names.
    """
    assert acetone.has_unique_atom_names is True


def test_make_unique_names():
    """
    After loading a molecule with non unique atom names make sure a unique set is automatically generated.
    """
    # load the molecule with missing names
    mol = Ligand.from_file(get_data("missing_names.pdb"))
    # make sure they have been converted
    assert mol.has_unique_atom_names is True


def test_bonds(acetone):
    """Make sure after reading a molecule file that we have the correct set of bonds.
    This is done by comparing back to the rdkit molecule.
    """
    reference = [(0, 4), (0, 5), (0, 6), (0, 1), (1, 2), (1, 3), (3, 7), (3, 8), (3, 9)]
    # make sure all bonds present
    for bond in acetone.bonds:
        assert bond in reference
    # make sure the number of bonds is the same
    assert len(reference) == acetone.n_bonds


def test_angles(acetone):
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
    for angle in acetone.angles:
        assert angle in reference

    assert len(reference) == acetone.n_angles


def test_dihedrals(acetone):
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

    for bond, dihedrals in acetone.dihedrals.items():
        for torsion in dihedrals:
            assert torsion in reference[bond]
        assert len(dihedrals) == len(reference[bond])

    # now check the total number of dihedrals
    assert acetone.n_dihedrals == 12


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


def test_impropers(acetone):
    """
    Make sure all improper torsions can be found in acetone.
    note this is fragile as the definition of improper may change.
    """
    reference = [(1, 0, 2, 3)]
    for improper in acetone.improper_torsions:
        assert improper in reference

    assert acetone.n_improper_torsions == len(reference)


def test_coords(acetone):
    """
    Make sure that when reading an input file the coordinates are saved.
    """
    assert acetone.coords["input"].shape == (acetone.n_atoms, 3)


def test_bond_lengths(acetone):
    """
    Make sure we can measure bond lengths for a given conformer and the distances match those given by rdkit.
    """
    bond_lengths = acetone.measure_bonds(input_type="input")
    rdkit_mol = acetone.rdkit_mol
    for bond, length in bond_lengths.items():
        assert pytest.approx(
            rdMolTransforms.GetBondLength(rdkit_mol.GetConformer(), *bond) == length
        )


def test_measure_angles(acetone):
    """
    Make sure we can correctly measure all of the angles in the molecule.
    """
    angle_values = acetone.measure_angles(input_type="input")
    rdkit_mol = acetone.rdkit_mol
    for angle, value in angle_values.items():
        assert pytest.approx(
            rdMolTransforms.GetAngleDeg(rdkit_mol.GetConformer(), *angle) == value
        )


def test_measure_dihedrals(acetone):
    """
    Make sure we can correctly measure all dihedrals in the molecule.
    """
    dihedral_values = acetone.measure_dihedrals(input_type="input")
    rdkit_mol = acetone.rdkit_mol
    for dihedral, value in dihedral_values.items():
        assert pytest.approx(
            rdMolTransforms.GetDihedralDeg(rdkit_mol.GetConformer(), *dihedral) == value
        )


def test_measure_no_dihedrals():
    """Make sure None is returned when there are no dihedrals to measure."""
    mol = Ligand(get_data("water.pdb"))
    assert mol.measure_dihedrals(input_type="input") is None


def test_get_atom(acetone):
    """
    Make sure we can pull the desired atom.
    """
    for atom in acetone.atoms:
        q_atom = acetone.get_atom_with_name(name=atom.atom_name)
        assert q_atom == atom


def test_get_atom_missing(acetone):
    """
    If we cannot find an atom with this name, make sure to throw an error.
    """
    with pytest.raises(AttributeError):
        acetone.get_atom_with_name(name="test")


def test_n_atoms(acetone):
    """
    Make sure all atoms are loaded into the system correctly.
    """
    assert acetone.n_atoms == 10


def test_rotatable_bonds_filtered(acetone):
    """
    For acetone while there are dihedrals we do not class these as rotatable as they are methyl, make sure
    this is true.
    """
    assert acetone.rotatable_bonds is None
    assert acetone.n_rotatable_bonds == 0


def test_no_rotatable_bonds():
    """
    If there are no dihedrals in the molecule make sure we return None.
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


def test_atom_types(acetone):
    """
    Make sure we can assign atom types to a molecule based on the CIP rank.
    For acetone we epect the following types
    - all hydrogen
    - methyl carbons
    - carbonyl carbon
    - carbonyl oxygen
    """
    atom_types = acetone.atom_types
    unique_types = set(list(atom_types.values()))
    assert len(unique_types) == 4


def test_bond_types(acetone):
    """
    Make sure based on the given atom types we can create bond types.
    """
    bond_types = acetone.bond_types
    unique_types = set(list(bond_types.keys()))
    assert len(unique_types) == 3


def test_angle_types(acetone):
    """
    Make sure based on the given atom types that we can create angle types.
    """
    angle_types = acetone.angle_types
    unique_types = set(list(angle_types.keys()))
    assert len(unique_types) == 4


def test_dihedral_types(acetone):
    """
    Make sure based on the given atom types we can create dihedral types.
    """
    dihedral_types = acetone.dihedral_types
    unique_types = set(list(dihedral_types.keys()))
    assert len(unique_types) == 2


def test_improper_types(acetone):
    """
    Make sure based on the given atom types we can create improper types.
    """
    improper_types = acetone.improper_types
    unique_types = set(list(improper_types.keys()))
    assert len(unique_types) == 1


def test_repr():
    """Make sure the ligand repr works."""
    mol = Ligand(get_data("water.pdb"))
    repr(mol)


@pytest.mark.parametrize(
    "trunc", [pytest.param(True, id="Trunc"), pytest.param(False, id="No trunc")]
)
def test_str(trunc):
    """
    Make sure that the ligand str method does not raise an error.
    """
    mol = Ligand(get_data("water.pdb"))
    mol.__str__(trunc=trunc)


def test_to_openmm_coords(acetone):
    """
    Make sure we can convert the coordinates to openmm style coords
    """
    coords = acetone.openmm_coordinates(input_type="input")
    assert np.allclose(coords[0] * 10, acetone.coords["input"])


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


def test_pickle_round_trip(tmpdir, acetone):
    """
    test dumping a molecule to pickle and loading it back in.
    """
    with tmpdir.as_cwd():
        acetone.pickle(state="test")
        mols = unpickle()
        pickle_mol = mols["test"]
        for atom in acetone.atoms:
            pickle_atom = pickle_mol.get_atom_with_name(atom.atom_name)
            assert pickle_atom.__dict__ == atom.__dict__
        assert acetone.bonds == pickle_mol.bonds
        assert acetone.angles == pickle_mol.angles
        assert acetone.dihedrals == pickle_mol.dihedrals


def test_double_pickle(tmpdir, acetone):
    """
    Make sure we can add multiple pickled objects to the same file.
    """
    with tmpdir.as_cwd():
        acetone.pickle(state="input")
        # remove all coords
        acetone.coords["input"] = []
        acetone.pickle(state="after")

        # now check we have both states
        mols = unpickle()
        assert "after" in mols
        assert "input" in mols


def test_write_xyz_single_conformer(tmpdir, acetone):
    """
    Write a single conformer xyz file for the molecule.
    """
    with tmpdir.as_cwd():
        acetone.write_xyz(input_type="input", name="acetone")

        # now read in and check the file
        with open("acetone.xyz") as xyz:
            lines = xyz.readlines()
            # atoms plus 2 comment lines
            assert len(lines) == acetone.n_atoms + 2
            assert float(lines[0]) == acetone.n_atoms
            # now loop over the atoms and make sure they match and the coords
            for i, line in enumerate(lines[2:]):
                atom = acetone.atoms[i]
                assert atom.atomic_symbol == line.split()[0]
                assert np.allclose(
                    acetone.coords["input"][i],
                    [float(coord) for coord in line.split()[1:]],
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


def test_pdb_round_trip(tmpdir, acetone):
    """
    Make sure we can write a molecule to pdb and load it back.
    """
    with tmpdir.as_cwd():
        acetone.write_pdb(input_type="input", name="test")

        mol2 = Ligand("test.pdb")
        for atom in acetone.atoms:
            pickle_atom = mol2.get_atom_with_name(atom.atom_name)
            assert pickle_atom.__dict__ == atom.__dict__
        assert acetone.bonds == mol2.bonds
        assert acetone.angles == mol2.angles
        assert acetone.dihedrals == mol2.dihedrals


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("acetone.pdb", id="pdb"),
        pytest.param("acetone.mol2", id="mol2"),
        pytest.param("bace0.mol", id="mol"),
        pytest.param("bace0.sdf", id="sdf"),
    ],
)
def test_ligand_from_file(file_name):
    """
    For the given file type make sure rdkit can parse it and return the molecule.
    """
    mol = Ligand.from_file(file_name=get_data(file_name))
    assert mol.n_atoms > 1
    assert mol.n_bonds > 1
    assert mol.name is not None


def test_ligand_file_missing():
    """
    Make sure that if the file is missing we get an error.
    """
    with pytest.raises(FileNotFoundError):
        _ = Ligand.from_file(file_name="test.pdb")


def test_ligand_file_not_supported():
    """
    Make sure we raise an error when an unsupported file type is passed.
    """
    with pytest.raises(FileTypeError):
        _ = Ligand(get_data("bace0.xyz"))


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("bace0.mol", id="mol"),
        pytest.param("bace0.sdf", id="sdf"),
        pytest.param("bace0.xyz", id="xyz"),
    ],
)
def test_add_conformers(file_name):
    """
    Load up the bace pdb and then add conformers to it from other file types.
    """
    mol = Ligand.from_file(file_name=get_data("bace0.pdb"))
    mol.add_conformers(file_name=get_data(file_name), input_type="mm")
    assert np.allclose(mol.coords["input"], mol.coords["mm"])


@pytest.mark.parametrize(
    "smiles",
    [
        pytest.param("C", id="No hydrogens"),
        pytest.param("[H]C([H])([H])[H]", id="Explicit hydrogens"),
    ],
)
def test_from_smiles(smiles):
    """
    Make sure hydrogens are added to a molecule when needed.
    """
    mol = Ligand.from_smiles(smiles_string=smiles, name="methane")
    # count the number of hydrogens
    hs = sum([1 for atom in mol.atoms if atom.atomic_symbol == "H"])
    assert hs == 4


def test_from_rdkit():
    """
    Make sure we can create a molecule directly from an rdkit object.
    """
    # load a molecule with openff
    offmol = OFFMolecule.from_file(file_path=get_data("bace0.sdf"))
    # make a ligand from the openff object
    mol = Ligand.from_rdkit(rdkit_mol=offmol.to_rdkit())
    # make sure we have the same molecule
    mol2 = Ligand.from_file(get_data("bace0.sdf"))
    for i in range(mol.n_atoms):
        atom1 = mol.atoms[i]
        atom2 = mol2.atoms[i]
        assert atom1.__dict__ == atom2.__dict__

    assert np.allclose(mol.coords["input"], mol2.coords["input"])
