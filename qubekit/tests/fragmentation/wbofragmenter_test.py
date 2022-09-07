from qubekit.molecules import Ligand
from qubekit.fragmentation import WBOFragmenter


def test_fragment_butane(tmpdir):
    """
    Fragmenter is running? Sanity check with butane
    """
    butane = Ligand.from_smiles('CCCC', 'butane')

    with tmpdir.as_cwd():
        fragmenter = WBOFragmenter()
        ligand = fragmenter.run(butane)

        assert len(ligand.fragments) == 1


def test_fragment_deduplication_bace(tmpdir, bace_fragmented):
    """
    Test if fragment deduplication merges the same fragments together
    """
    # The WBO fragmenter creates 3 fragments
    assert len(bace_fragmented.fragments) == 3

    # run deduplication
    fragmenter = WBOFragmenter()
    bace_fragmented.fragments = fragmenter._deduplicate_fragments(bace_fragmented.fragments)

    # two fragments are the same and therefore are merged
    assert len(bace_fragmented.fragments) == 2
    # because two fragments are merged, one has two torsions for scanning
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
