from qubekit.fragmentation import WBOFragmentation
from qubekit.molecules import Ligand


def test_fragment_butane(tmpdir):
    """
    Fragmenter is running? Sanity check with butane
    """
    butane = Ligand.from_smiles("CCCC", "butane")

    with tmpdir.as_cwd():
        fragmenter = WBOFragmentation()
        ligand = fragmenter.run(butane)

        assert len(ligand.fragments) == 1


def test_fragment_deduplication_bace(tmpdir, bace_fragmented):
    """
    Test if fragment deduplication merges the same fragments together
    """
    # The WBO fragmenter creates 3 fragments
    assert len(bace_fragmented.fragments) == 3

    # run deduplication
    fragmenter = WBOFragmentation()
    bace_fragmented.fragments = fragmenter._deduplicate_fragments(
        bace_fragmented.fragments
    )

    # two fragments are the same and therefore are merged
    assert len(bace_fragmented.fragments) == 2
    # because two fragments are merged, one has two torsions for scanning
    assert any([len(frag.bond_indices) == 2 for frag in bace_fragmented.fragments])


def test_fragment_deduplication_symmetry_equivalent(symmetry_fragments):
    """Make sure we can deduplicate symmetry equivalent torsions across two different molecules"""

    assert len(symmetry_fragments.fragments) == 2
    assert all(
        [len(fragment.bond_indices) == 1 for fragment in symmetry_fragments.fragments]
    )
    fragmenter = WBOFragmentation()
    deduplicated_fragments = fragmenter._deduplicate_fragments(
        fragments=symmetry_fragments.fragments
    )
    # we should have one molecule with one scan
    assert len(deduplicated_fragments) == 1
    assert len(deduplicated_fragments[0].bond_indices) == 1


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
