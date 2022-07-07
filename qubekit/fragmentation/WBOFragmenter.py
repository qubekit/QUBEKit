from typing_extensions import Literal
from typing import List, Optional, Union

from pydantic import Field
from qcelemental.util import which_import

from qubekit.molecules import Ligand
from qubekit.utils.datastructures import StageBase


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

        bespokefit = which_import(
            "openff.bespokefit",
            return_bool=True,
            raise_error=True,
            raise_msg="Please install via `conda install -c conda-forge openff-bespokefit`.",
        )

        return fragmenter and bespokefit

    def run(self, molecule: Ligand, **kwargs) -> Ligand:
        from openff.toolkit.topology import Molecule as OFFMolecule

        # from openff.bespokefit.executor.services.fragmenter.worker import _deduplicate_fragments
        from openff.fragmenter.depiction import depict_fragmentation_result
        from openff.fragmenter.fragment import WBOFragmenter

        fragmenter = WBOFragmenter()

        # convert to an OpenFF Molecule
        mapped_smiles = molecule.to_smiles(mapped=True)
        off_molecule: OFFMolecule = OFFMolecule.from_mapped_smiles(mapped_smiles)

        # fragment
        result = fragmenter.fragment(
            off_molecule, target_bond_smarts=self.rotatable_smirks
        )

        # deduplicated_result = _deduplicate_fragments(result).json()

        # simpler representation of the fragments
        molecule.fragments = {
            fragment.smiles: fragment.bond_indices for fragment in result.fragments
        }

        # visualise results in a file
        depict_fragmentation_result(result=result, output_file="fragments.html")

        return molecule

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
