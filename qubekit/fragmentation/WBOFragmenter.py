import itertools
from typing import List, TYPE_CHECKING
from typing_extensions import Literal

from pydantic import Field
from qcelemental.util import which_import

from qubekit.molecules import Ligand
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.fragmenter.fragment import (
        FragmentationResult,
    )


class WBOFragmenter(StageBase):

    type: Literal["WBOFragmenter"] = "WBOFragmenter"

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

    def run(self, molecule: Ligand, **kwargs) -> Ligand:
        from openff.toolkit.topology import Molecule as OFFMolecule

        from openff.fragmenter.depiction import depict_fragmentation_result
        from openff.fragmenter.fragment import WBOFragmenter

        fragmenter = WBOFragmenter()

        # convert to an OpenFF Molecule
        mapped_smiles = molecule.to_smiles(mapped=True)
        off_molecule: OFFMolecule = OFFMolecule.from_mapped_smiles(mapped_smiles)

        # fragment
        fragmentation = fragmenter.fragment(
            off_molecule, target_bond_smarts=self.rotatable_smirks
        )

        # remove duplicates
        fragmentation = self._deduplicate_fragments(fragmentation)

        # save the fragments in the molecule as ligand objects
        fragments = []
        for fragment in fragmentation.fragments:
            ligand = Ligand.from_smiles(fragment.smiles, "fragment")
            ligand.bond_indices = fragment.bond_indices
            fragments.append(ligand)
        molecule.fragments = fragments

        # visualise results in a file
        depict_fragmentation_result(result=fragmentation, output_file="fragments.html")

        return molecule

    def _deduplicate_fragments(
        self, fragmentation_result: "FragmentationResult"
    ) -> "FragmentationResult":
        """Remove symmetry equivalent fragments from the results."""
        unique_fragments = []

        # group by the same smiles
        for smile, fragments in itertools.groupby(
            fragmentation_result.fragments,
            lambda frag: frag.molecule.to_smiles(explicit_hydrogens=False),
        ):
            fragments = list(fragments)

            if len(fragments) == 1:
                unique_fragments.extend(fragments)
                continue

            symmetry_groups = set()
            for fragment in fragments:
                bond_map = fragment.bond_indices
                fragment_mol = fragment.molecule
                # get the index of the atoms in the fragment
                atom1, atom2 = self._get_atom_index(
                    fragment_mol, bond_map[0]
                ), self._get_atom_index(fragment_mol, bond_map[1])
                symmetry_classes = self._rd_get_atom_symmetries(fragment_mol)
                symmetry_group = tuple(
                    sorted([symmetry_classes[atom1], symmetry_classes[atom2]])
                )
                if symmetry_group not in symmetry_groups:
                    symmetry_groups.add(symmetry_group)
                    unique_fragments.append(fragment)

        fragmentation_result.fragments = unique_fragments

        return fragmentation_result

    def _rd_get_atom_symmetries(self, molecule: "Molecule") -> List[int]:

        from rdkit import Chem

        rd_mol = molecule.to_rdkit()
        return list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))

    def _get_atom_index(self, molecule: "Molecule", map_index: int) -> int:
        """Returns the atom index of the atom in a molecule which has the specified map
        index.

        Parameters
        ----------
        molecule
            The molecule containing the atom.
        map_index
            The map index of the atom in the molecule.

        Returns
        -------
            The corresponding atom index
        """
        inverse_atom_map = {
            j: i for i, j in molecule.properties.get("atom_map", {}).items()
        }

        atom_index = inverse_atom_map.get(map_index, None)

        assert (
            atom_index is not None
        ), f"{map_index} does not correspond to an atom in the molecule."

        return atom_index

    def start_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that stage is starting with any important options.
        """
        # fixme - check if the fragmenter uses the same default smirks?
        return "WBOFragmenter is starting the job"

    def finish_message(self, **kwargs) -> str:
        """
        A friendly message to let users know that the stage is complete and any checks that have been performed.
        """
        return f"WBOFragmenter has ended"
