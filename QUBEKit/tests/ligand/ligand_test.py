import networkx as nx
import numpy as np
import pytest
from openff.toolkit.topology import Molecule as OFFMolecule
from rdkit.Chem import rdMolTransforms
from simtk import unit

from QUBEKit.ligand import Ligand
from QUBEKit.utils.exceptions import FileTypeError, SmartsError, TopologyMismatch
from QUBEKit.utils.file_handling import get_data
from QUBEKit.utils.helpers import unpickle


@pytest.fixture
def acetone():
    """
    Make a ligand class from the acetone pdb.
    """
    return Ligand.from_file(file_name=get_data("acetone.sdf"))


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
        assert (bond.atom1_index, bond.atom2_index) in reference
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
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
    assert mol.dihedrals is None
    assert mol.n_dihedrals == 0


def test_no_impropers():
    """
    Make sure we return None when no impropers are found in the molecule.
    """
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
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
    assert acetone.coordinates.shape == (acetone.n_atoms, 3)


def test_bond_lengths(acetone):
    """
    Make sure we can measure bond lengths for a given conformer and the distances match those given by rdkit.
    """
    bond_lengths = acetone.measure_bonds()
    rdkit_mol = acetone.to_rdkit()
    for bond, length in bond_lengths.items():
        assert pytest.approx(
            rdMolTransforms.GetBondLength(rdkit_mol.GetConformer(), *bond) == length
        )


def test_measure_angles(acetone):
    """
    Make sure we can correctly measure all of the angles in the molecule.
    """
    angle_values = acetone.measure_angles()
    rdkit_mol = acetone.to_rdkit()
    for angle, value in angle_values.items():
        assert pytest.approx(
            rdMolTransforms.GetAngleDeg(rdkit_mol.GetConformer(), *angle) == value
        )


def test_measure_dihedrals(acetone):
    """
    Make sure we can correctly measure all dihedrals in the molecule.
    """
    dihedral_values = acetone.measure_dihedrals()
    rdkit_mol = acetone.to_rdkit()
    for dihedral, value in dihedral_values.items():
        assert pytest.approx(
            rdMolTransforms.GetDihedralDeg(rdkit_mol.GetConformer(), *dihedral) == value
        )


def test_measure_no_dihedrals():
    """Make sure None is returned when there are no dihedrals to measure."""
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
    assert mol.measure_dihedrals() is None


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
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
    assert mol.rotatable_bonds is None
    assert mol.n_rotatable_bonds == 0


def test_rotatable_bonds():
    """
    Make sure we can find true rotatable bonds for a molecule.
    """
    mol = Ligand.from_file(file_name=get_data("biphenyl.pdb"))
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
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
    repr(mol)


@pytest.mark.parametrize(
    "trunc", [pytest.param(True, id="Trunc"), pytest.param(False, id="No trunc")]
)
def test_str(trunc):
    """
    Make sure that the ligand str method does not raise an error.
    """
    mol = Ligand.from_file(file_name=get_data("water.pdb"))
    mol.__str__(trunc=trunc)


def test_to_openmm_coords(acetone):
    """
    Make sure we can convert the coordinates to openmm style coords
    """
    coords = acetone.openmm_coordinates()
    assert np.allclose(coords.in_units_of(unit.angstrom), acetone.coordinates)


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
        acetone.coordinates = None
        acetone.pickle(state="after")

        # now check we have both states
        mols = unpickle()
        assert "after" in mols
        assert "input" in mols


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("acetone.xyz", id="xyz"),
        pytest.param("acetone.pdb", id="pdb"),
        pytest.param("acetone.sdf", id="sdf"),
        pytest.param("acetone.mol", id="mol"),
    ],
)
def test_to_file(tmpdir, file_name, acetone):
    """
    Try and write out a molecule to the specified file type.
    """
    with tmpdir.as_cwd():
        acetone.to_file(file_name=file_name)


def test_to_file_fail(tmpdir, acetone):
    """
    Make sure an error is raised if we try and write to an unsupported file type.
    """
    with tmpdir.as_cwd():
        with pytest.raises(FileTypeError):
            acetone.to_file(file_name="badfile.smi")


def test_write_xyz_multiple_conformer(tmpdir):
    """
    Make sure we can write multiple conformer xyz files for a molecule.
    """
    with tmpdir.as_cwd():
        mol = Ligand.from_file(file_name=get_data("butane.pdb"))
        # fake a set of conformers
        coords = [mol.coordinates, np.random.random((mol.n_atoms, 3))]
        mol.to_multiconformer_file(file_name="butane.xyz", positions=coords)

        # now read in the file again
        with open("butane.xyz") as xyz:
            lines = xyz.readlines()
            assert len(lines) == 2 * mol.n_atoms + 4


@pytest.mark.parametrize(
    "file_name",
    [
        pytest.param("acetone.pdb", id="pdb"),
        pytest.param("acetone.sdf", id="sdf"),
        pytest.param("acetone.xyz", id="xyz"),
    ],
)
def test_write_multi_conformer(tmpdir, acetone, file_name):
    """
    Make sure the each file type is supported.
    """
    with tmpdir.as_cwd():
        coords = [acetone.coordinates, np.random.random((acetone.n_atoms, 3))]
        acetone.to_multiconformer_file(file_name=file_name, positions=coords)


def test_sdf_round_trip(tmpdir, acetone):
    """
    Make sure we can write a molecule to pdb and load it back.
    """
    with tmpdir.as_cwd():
        acetone.to_file(file_name="test.sdf")

        mol2 = Ligand.from_file(file_name="test.sdf")
        for atom in acetone.atoms:
            pickle_atom = mol2.get_atom_with_name(atom.atom_name)
            assert pickle_atom.dict() == atom.dict()
        for i in range(acetone.n_bonds):
            assert acetone.bonds[i].dict() == mol2.bonds[i].dict()
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
        _ = Ligand.from_file(file_name=get_data("bace0.xyz"))


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
    mol.coordinates = None
    mol.add_conformer(file_name=get_data(file_name))
    assert mol.coordinates.shape == (mol.n_atoms, 3)


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
        assert atom1.dict() == atom2.dict()
    for i in range(mol.n_bonds):
        bond1 = mol.bonds[i]
        bon2 = mol2.bonds[i]
        assert bond1.dict() == bon2.dict()

    assert np.allclose(mol.coordinates, mol2.coordinates)


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param("bace0.sdf", id="bace0 chiral"),
        pytest.param("12-dichloroethene.sdf", id="12dichloroethene bond stereo"),
    ],
)
def test_to_rdkit(molecule):
    """
    Make sure we can convert to rdkit.
    We test on bace which has a chiral center and 12-dichloroethene which has a stereo bond.
    """
    from rdkit import Chem

    mol = Ligand.from_file(file_name=get_data(molecule))
    rd_mol = mol.to_rdkit()
    # make sure the atom and bond stereo match
    for atom in rd_mol.GetAtoms():
        qb_atom = mol.atoms[atom.GetIdx()]
        assert atom.GetIsAromatic() is qb_atom.aromatic
        if qb_atom.stereochemistry is not None:
            if qb_atom.stereochemistry == "S":
                assert atom.GetChiralTag() == Chem.CHI_TETRAHEDRAL_CCW
            else:
                assert atom.GetChiralTag() == Chem.CHI_TETRAHEDRAL_CW
    for bond in rd_mol.GetBonds():
        qb_bond = mol.bonds[bond.GetIdx()]
        assert qb_bond.aromatic is bond.GetIsAromatic()
        assert qb_bond.bond_order == bond.GetBondTypeAsDouble()
        if qb_bond.stereochemistry is not None:
            if qb_bond.stereochemistry == "E":
                assert bond.GetStereo() == Chem.BondStereo.STEREOE
            else:
                assert bond.GetStereo() == Chem.BondStereo.STEREOZ


@pytest.mark.parametrize(
    "molecule, charge",
    [
        pytest.param("acetone.sdf", 0, id="acetone"),
        pytest.param("bace0.sdf", 1, id="bace0"),
        pytest.param("pyridine.sdf", 0, id="pyridine"),
    ],
)
def test_charge(molecule, charge):
    """
    Make sure that the charge is correctly identified.
    """
    mol = Ligand.from_file(file_name=get_data(molecule))
    assert mol.charge == charge


@pytest.mark.parametrize(
    "molecule",
    [
        pytest.param("pyridine.sdf", id="pyridine"),
        pytest.param("acetone.sdf", id="acetone"),
        pytest.param("bace0.sdf", id="bace0"),
    ],
)
def test_to_topology(molecule):
    """
    Make sure that a topology generated using qubekit matches an openff one.
    """
    mol = Ligand.from_file(file_name=get_data(molecule))
    offmol = OFFMolecule.from_file(file_path=get_data(molecule))
    assert (
        nx.algorithms.isomorphism.is_isomorphic(mol.to_topology(), offmol.to_networkx())
        is True
    )


def test_to_smiles_isomeric():
    """
    Make sure we can write out smiles strings with the correct settings.
    """
    # use bace as it has a chiral center
    mol = Ligand.from_file(file_name=get_data("bace0.sdf"))
    smiles = mol.to_smiles(isomeric=True, explicit_hydrogens=False, mapped=False)
    assert "@@" in smiles
    smiles = mol.to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False)
    assert "@" not in smiles


def test_to_smiles_hydrogens(acetone):
    """
    Make sure the explicit hydrogens flag is respected.
    """
    smiles_h = acetone.to_smiles(isomeric=False, explicit_hydrogens=True, mapped=False)
    assert smiles_h == "[H][C]([H])([H])[C](=[O])[C]([H])([H])[H]"
    smiles = acetone.to_smiles(isomeric=False, explicit_hydrogens=False, mapped=False)
    assert smiles == "CC(C)=O"


def test_to_mapped_smiles():
    """
    Make sure the the mapped smiles flag is respected.
    """
    mol = Ligand.from_file(file_name=get_data("bace0.sdf"))
    no_map = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=False)
    mapped = mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
    assert no_map != mapped


def test_smarts_matches(acetone):
    """
    Make sure we can find the same environment matches as openff.
    """
    matches = acetone.get_smarts_matches(smirks="[#6:1](=[#8:2])-[#6]")
    # make sure the atoms are in the correct order
    assert len(matches) == 1
    match = matches[0]
    assert acetone.atoms[match[0]].atomic_symbol == "C"
    assert acetone.atoms[match[1]].atomic_symbol == "O"
    # make sure these atoms are bonded
    _ = acetone.get_bond_between(*match)
    off = OFFMolecule.from_file(file_path=get_data("acetone.sdf"))
    off_matches = off.chemical_environment_matches(query="[#6:1](=[#8:2])-[#6]")
    # check we match the same bonds
    assert set(off_matches) == set(matches)


def test_smarts_matches_bad_query(acetone):
    """
    Make sure an error is raised if we try and search with a bad smarts pattern.
    """
    with pytest.raises(SmartsError):
        acetone.get_smarts_matches(smirks="skfbj")


def test_smarts_no_matches(acetone):
    """
    Make sure None is returned if we have no matches.
    """
    matches = acetone.get_smarts_matches(smirks="[#16:1]")
    assert matches is None


def test_get_bond_error(acetone):
    """
    Make sure an error is raised if we can not find a bond between to atoms.
    """
    with pytest.raises(TopologyMismatch):
        acetone.get_bond_between(atom1_index=4, atom2_index=9)
