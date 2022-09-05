from typing import TYPE_CHECKING, List, Optional

from openmm import System, app
from typing_extensions import Literal

from qubekit.parametrisation.base_parametrisation import Parametrisation

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class Gromacs(Parametrisation):
    """
    Parametrise a molecule using gromacs input files via OpenMM
    """

    type: Literal["Gromacs"] = "Gromacs"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def _improper_torsion_ordering(cls) -> str:
        return "amber"

    def _build_system(
        self, molecule: "Ligand", input_files: Optional[List[str]] = None
    ) -> System:

        # get the topfile
        top_file = None
        if input_files is not None:
            for file in input_files:
                if file.endswith(".top"):
                    top_file = file
                    break
        # if it was not given try and guess it
        top_file = top_file or f"{molecule.name}.top"

        top = app.GromacsTopFile(
            top_file,
        )
        system = top.createSystem()
        return system
