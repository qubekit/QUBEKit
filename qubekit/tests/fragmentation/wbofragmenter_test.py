from qubekit.molecules import Ligand
from qubekit.fragmentation import WBOFragmenter


def test_generate_fragments():
    """
    Generate fragments for the example case Cobimetinib
    """
    Cobimetinib = "OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)[C@@H]1CCCCN1"
    molecule = Ligand.from_smiles(Cobimetinib, "Cobimetinib")

    fragmenter = WBOFragmenter()
    result = fragmenter.run(molecule)

    # Cobimetinib has 5 fragments according to the example
    # https://docs.openforcefield.org/projects/fragmenter/en/latest/fragment-molecules.html
    assert len(result.fragments) == 5
