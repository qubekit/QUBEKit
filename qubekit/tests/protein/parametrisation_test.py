import shutil

import pytest

from qubekit.molecules import Protein
from qubekit.parametrisation import XMLProtein
from qubekit.utils.file_handling import get_data


def test_protein_params(tmpdir):
    """
    Load up the small protein and make sure it is parametrised with the general qube forcefield.
    """
    with tmpdir.as_cwd():
        pro = Protein.from_file(file_name=get_data("capped_leu.pdb"))
        shutil.copy(get_data("capped_leu.pdb"), "capped_leu.pdb")
        xml_pro = XMLProtein()
        new_pro = xml_pro.parametrise_molecule(molecule=pro)
        new_pro.write_parameters("pro.xml")
