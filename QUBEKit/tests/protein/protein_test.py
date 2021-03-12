import pytest

from QUBEKit.molecules import Protein
from QUBEKit.utils.exceptions import FileTypeError
from QUBEKit.utils.file_handling import get_data


def test_from_pdb():
    """
    Make sure we can make a protein from a pdb file.
    """
    pro = Protein.from_file(file_name=get_data("capped_leu.pdb"))
    assert pro.n_atoms == 31
    assert pro.coordinates.shape == (pro.n_atoms, 3)


def test_normal_init():
    """
    Make sure a pdb file is directed to the from pdb path.
    """
    pro = Protein(get_data("capped_leu.pdb"))
    assert pro.n_atoms == 31


def test_not_pdb():
    """
    Make sure an error is raised if we try and make a protein from a non pdb file.
    """
    with pytest.raises(FileTypeError):
        _ = Protein.from_file(file_name=get_data("bace0.xyz"))


def test_pdb_round_trip(tmpdir):
    """
    Make sure we can round trip a protein to pdb.
    """
    with tmpdir.as_cwd():
        pro = Protein.from_file(file_name=get_data("capped_leu.pdb"))
        pro.write_pdb(name="test")

        pro2 = Protein.from_file("test.pdb")

        assert pro.n_atoms == pro2.n_atoms
        assert pro.n_bonds == pro.n_bonds
        for bond in pro.bonds:
            assert bond in pro2.bonds
        for angle in pro.angles:
            assert angle in pro2.angles
        for central_bond, torsions in pro.dihedrals.items():
            for d in torsions:
                assert d in pro2.dihedrals[central_bond]
