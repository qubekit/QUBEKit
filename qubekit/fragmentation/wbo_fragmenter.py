import itertools
from typing import TYPE_CHECKING, List

from openff.fragmenter.depiction import depict_fragmentation_result
from openff.fragmenter.fragment import WBOFragmenter
from openff.toolkit.topology import Molecule as OFFMolecule
from pydantic import Field
from qcelemental.util import which_import
from typing_extensions import Literal

from qubekit.molecules import Fragment, Ligand
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from openff.fragmenter.fragment import FragmentationResult


class WBOFragmentation(StageBase, WBOFragmenter):
    type: Literal["WBOFragmentation"] = "WBOFragmentation"

    rotatable_smirks: List[str] = Field(
        ["[!#1]~[!$(*#*)&!D1:1]-,=;!@[!$(*#*)&!D1:2]~[!#1]"],
        description="The rotatable smirks to use for fragmentation",
    )

    @classmethod
    def is_available(cls) -> bool:
        fragmenter = which_import(
            "openff.fragmenter",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install -c conda-forge openff-fragmenter`.",
        )

        rdkit = which_import(
            "rdkit",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install -c conda-forge rdkit`.",
        )

        return fragmenter and rdkit

    def _results_to_ligand(
        self, input_ligand: Ligand, fragmentation_result: "FragmentationResult"
    ) -> Ligand:
        """
        Repackage the fragmentation results into our data types. Make sure to deduplicate the parent as well.
        """
        molecule = Ligand.from_smiles(
            fragmentation_result.parent_smiles, input_ligand.name
        )
        fragments = []
        for i, wbo_fragment in enumerate(fragmentation_result.fragments):
            fragment = Fragment.from_smiles(wbo_fragment.smiles, f"fragment_{i}")

            # attach the fragments as their own ligands
            fragment.bond_indices.append(wbo_fragment.bond_indices)
            if fragment != molecule:
                # only save if the fragment is not the parent to avoid duplicated hessian/charge calculations
                fragments.append(fragment)
            else:
                molecule.bond_indices.extend(fragment.bond_indices)

        molecule.fragments = fragments

        return molecule

    def run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        """Overwrite base so we don't try and fragment fragments on restarts."""
        return self._run(molecule, *args, **kwargs)

    def _run(self, molecule: Ligand, *args, **kwargs) -> Ligand:
        # convert to an OpenFF Molecule
        off_molecule: OFFMolecule = OFFMolecule.from_smiles(
            molecule.to_smiles(), allow_undefined_stereo=True
        )

        fragmentation_result = self.fragment(
            off_molecule, target_bond_smarts=self.rotatable_smirks
        )

        depict_fragmentation_result(
            result=fragmentation_result, output_file="fragments.html"
        )

        ligand = self._results_to_ligand(
            input_ligand=molecule, fragmentation_result=fragmentation_result
        )

        ligand.fragments = self._deduplicate_fragments(ligand.fragments) or None

        return ligand

    @staticmethod
    def _group_subfragments_together(fragments):
        # modifies the fragments
        # some fragments are subfragments of other fragments, group together
        # sort first by size, largest first
        # f.to_rdkit().HasSubstructMatch()
        sorted_frags = sorted(fragments, key=lambda f: len(f.atoms), reverse=True)
        from rdkit import Chem

        substructures = []
        for i, f_larger in enumerate(sorted_frags):
            # if this fragment already is a substructure of another fragment,
            # then it does not matter that it is a superstructure to another fragment
            if i in substructures:
                continue

            for j, f_smaller in enumerate(sorted_frags[i+1:], start=i+1):
                if Chem.RemoveHs(f_larger.to_rdkit()).HasSubstructMatch(Chem.RemoveHs(f_smaller.to_rdkit())):
                    # avoid considering this fragment as a parent to another fragment
                    substructures.append(j)

                    # merge the smaller structure into the larger structure
                    f_larger.bond_indices.extend(f_smaller.bond_indices)

        # return fragments without the substructures
        fragments_without_substructures = [fragment for i, fragment in enumerate(sorted_frags)
                                           if i not in substructures]
        return fragments_without_substructures

    def _deduplicate_fragments(
        self, original_fragments: List["Fragment"]
    ) -> List["Fragment"]:
        """
        Remove symmetry equivalent fragments from the results. If a bond is the same and appears twice,
        it has to be computed only once.

        Where fragments are the same but they focus on different bonds, combine them and keep all bonds.
        """
        unique_fragments = []

        sorted_fragments = sorted(
            original_fragments,
            key=lambda frag: frag.to_smiles(explicit_hydrogens=False),
        )

        # group fragments by smiles
        for smile, fragments in itertools.groupby(
            sorted_fragments,
            lambda frag: frag.to_smiles(explicit_hydrogens=False),
        ):
            fragments = list(fragments)

            if len(fragments) == 1:
                unique_fragments.extend(fragments)
                continue

            # filter out symmetrical bonds
            symmetry_groups = set()
            fragments_assymetry = []
            for fragment in fragments:
                bond_map = fragment.bond_indices
                a1, a2 = [a for a in fragment.atoms if a.map_index in bond_map[0]]

                symmetry_classes = fragment.atom_types
                symmetry_group = frozenset(
                    {
                        int(symmetry_classes[a1.atom_index]),
                        int(symmetry_classes[a2.atom_index]),
                    }
                )

                if symmetry_group not in symmetry_groups:
                    symmetry_groups.add(symmetry_group)
                    fragments_assymetry.append(fragment)

            # the smiles are the same so
            # merge into one fragment to compute hessian/charges/etc only once
            # but keep all the bonds that need scanning
            fragments_assymetry[0].bond_indices = [
                f.bond_indices[0] for f in fragments_assymetry
            ]
            unique_fragments.append(fragments_assymetry[0])

        # check if the fragments are substructures of other fragmetns, if they are, use the larger fragments instead,
        # and add to them the bond_indices from the smaller fragments,
        grouped_fragments = self._group_subfragments_together(unique_fragments)

        # re number the unique fragments
        for i, fragment in enumerate(grouped_fragments):
            fragment.name = f"fragment_{i}"

        return grouped_fragments

    def start_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that stage is starting with any important options.
        """

        return "Fragmenting molecule using the WBOFragmenter"

    def finish_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that the stage is complete and any checks that have been performed.
        """
        return f"Molecule produced {kwargs['molecule'].n_fragments} fragments"
