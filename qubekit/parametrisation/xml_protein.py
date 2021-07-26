from typing import TYPE_CHECKING, List, Optional

from simtk.openmm import System, app
from typing_extensions import Literal

from qubekit.parametrisation.base_parametrisation import Parametrisation
from qubekit.proteins.protein_tools import qube_general

if TYPE_CHECKING:
    from qubekit.molecules import Protein


class XMLProtein(Parametrisation):
    """Read in the parameters for a proteins from the QUBEKit_general XML file and store them into the proteins."""

    type: Literal["XMLProtein"] = "XMLProtein"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def _improper_torsion_ordering(cls) -> str:
        return "default"

    def _build_system(
        self, molecule: "Protein", input_files: Optional[List[str]] = None
    ) -> System:
        """Serialise the input XML system using openmm."""

        # we can not to do residue conversion yet
        pdb = app.PDBFile(f"{molecule.name}.pdb")
        modeller = app.Modeller(pdb.topology, pdb.positions)

        # we only support the qube general xml
        xml = "QUBE_general_pi.xml"
        qube_general()
        forcefield = app.ForceField(xml)

        system = forcefield.createSystem(
            modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None
        )

        return system
