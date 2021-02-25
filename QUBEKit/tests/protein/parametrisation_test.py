from QUBEKit.ligand import Protein
from QUBEKit.parametrisation import XMLProtein
from QUBEKit.utils.file_handling import get_data
import shutil
import pytest


def test_protein_params(tmpdir):
    """
    Load up the small protein and make sure it is parametrised with the general qube forcefield.
    """
    with tmpdir.as_cwd():
        pro = Protein.from_file(file_name=get_data("capped_leu.pdb"))
        shutil.copy(get_data("capped_leu.pdb"), "capped_leu.pdb")
        XMLProtein(protein=pro)
