from qubekit.molecules import Ligand
from qubekit.fragmentation import WBOFragmenter


def test_generate_fragments(tmpdir):
    """
    Generate fragments for the example case Cobimetinib
    """
    Cobimetinib = "OC1(CN(C1)C(=O)C1=C(NC2=C(F)C=C(I)C=C2)C(F)=C(F)C=C1)[C@@H]1CCCCN1"
    molecule = Ligand.from_smiles(Cobimetinib, "Cobimetinib")

    with tmpdir.as_cwd():
        fragmenter = WBOFragmenter()
        ligand = fragmenter.run(molecule)

        # Cobimetinib has 5 fragments according to the example
        # https://docs.openforcefield.org/projects/fragmenter/en/latest/fragment-molecules.html
        assert len(ligand.fragments) == 5


def test_merging_fragments_bace(tmpdir, bace_fragmented):
    """
    Generate fragments for the example case BACE
    """
    # The fragmenter creates 3 fragments (see visualisation output), however
    # two of them are the same fragments and therefore they are merged.
    assert len(bace_fragmented.fragments) == 2
    assert any([len(frag.bond_indices) == 2 for frag in bace_fragmented.fragments])


def test_recovering_indices(tmpdir, bace_fragmented):
    """
    Check if you can find the original indices around which the fragments were built
    """
    for fragment in bace_fragmented.fragments:
        for pair in fragment.bond_indices:
            # find the indices of the correct parent atoms
            atom_indices = {
                a.atom_index for a in bace_fragmented.atoms if a.map_index in pair
            }
            assert len(atom_indices) == 2

            # verify that the two atoms are actually bounded
            assert any(
                {b.atom1_index, b.atom2_index} == atom_indices
                for b in bace_fragmented.bonds
            )
