from qubekit.fragmentation import WBOFragmentation
from qubekit.molecules import Ligand


def test_fragment(tmpdir):
    """
    Check the Fragmentation runs and returns the input with the Cl removed
    """
    biphen = Ligand.from_smiles(
        "C1=CC=C(C=C1)C2=CC(=CC=C2)Cl", "1-chloro-3-phenylbenzene"
    )

    with tmpdir.as_cwd():
        fragmenter = WBOFragmentation()
        ligand = fragmenter.run(biphen)

        assert len(ligand.fragments) == 1
        assert (
            ligand.fragments[0].to_smiles()
            == "[H][c]1[c]([H])[c]([H])[c](-[c]2[c]([H])[c]([H])[c]([H])[c]([H])[c]2[H])[c]([H])[c]1[H]"
        )


def test_fragmentation_same_molecule(tmpdir):
    """
    When fragmentation would return the same molecule, make sure it is deduplicated and the bond indices are transfered.
    """
    butane = Ligand.from_smiles("CCCC", "butane")
    with tmpdir.as_cwd():
        fragmenter = WBOFragmentation()
        ligand = fragmenter.run(molecule=butane)

        # the parent is the fragment so make sure it is removed
        assert ligand.fragments is None
        assert ligand.bond_indices == [(3, 4)]
        # make sure these are connected in the final molecule
        bond = [
            a.atom_index for a in ligand.atoms if a.map_index in ligand.bond_indices[0]
        ]
        _ = ligand.get_bond_between(*bond)


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
