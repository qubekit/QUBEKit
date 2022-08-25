import itertools
from typing import List, TYPE_CHECKING
from typing_extensions import Literal

from rdkit import Chem
from openff.fragmenter.fragment import WBOFragmenter
from openff.fragmenter.depiction import depict_fragmentation_result
from openff.toolkit.topology import Molecule as OFFMolecule
from pydantic import Field
from qcelemental.util import which_import

from qubekit.molecules import Ligand, Fragment
from qubekit.utils.datastructures import StageBase

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.fragmenter.fragment import (
        FragmentationResult,
    )


class WBOFragmenter(StageBase, WBOFragmenter):

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

    def _results_to_ligand(self, fragmentation):
        """
        TODO
        """
        # create the parent molecule
        molecule = Ligand.from_smiles(fragmentation.parent_smiles, 'ligand')

        for wbo_fragment in fragmentation.fragments:
            # partly verify that the fragments can still be mapped to the parent molecule
            atoms = [a for a in molecule.atoms if a.map_index in wbo_fragment.bond_indices]
            if len(atoms) != 2:
                raise Exception('The index used in the fragment to map to the parent could not be found in the parent.')

            fragment = Fragment.from_smiles(wbo_fragment.smiles, 'fragment')

            # attach the fragments as their own ligands
            fragment.bond_indices.append(wbo_fragment.bond_indices)

            molecule.fragments.append(fragment)

        return molecule

    def run(self, molecule: "Ligand", *args, **kwargs) -> "Ligand":
        return self._run(molecule, *args, **kwargs)

    def _run(self, molecule: Ligand, **kwargs) -> Ligand:
        # convert to an OpenFF Molecule
        off_molecule: OFFMolecule = OFFMolecule.from_smiles(molecule.to_smiles())

        fragmentation = self.fragment(
            off_molecule, target_bond_smarts=self.rotatable_smirks
        )

        # visualise results
        depict_fragmentation_result(result=fragmentation, output_file="fragments.html")

        # convert the results to Ligand class
        ligand = self._results_to_ligand(fragmentation)

        # remove duplicates
        # fixme - fragments with the same smiles should have a list of bond_indices instead
        fragmentation = self._deduplicate_fragments(ligand)

        return fragmentation

    def _deduplicate_fragments(
        self, ligand: "Ligand"
    ) -> "Ligand":
        """Remove symmetry equivalent fragments from the results."""
        unique_fragments = []

        # group fragments by smiles
        for smile, fragments in itertools.groupby(
            ligand.fragments,
            lambda frag: frag.to_smiles(explicit_hydrogens=False),
        ):
            fragments = list(fragments)

            if len(fragments) == 1:
                unique_fragments.extend(fragments)
                continue

            symmetry_groups = set()
            for fragment in fragments:
                bond_map = fragment.bond_indices
                a1, a2 = [a for a in fragment.atoms if a.map_index in bond_map[0]]

                symmetry_classes = fragment.atom_types
                symmetry_group = frozenset({symmetry_classes[a1.atom_index], symmetry_classes[a2.atom_index]})

                if symmetry_group not in symmetry_groups:
                    symmetry_groups.add(symmetry_group)
                    unique_fragments.append(fragment)

        ligand.fragments = unique_fragments

        return ligand

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
