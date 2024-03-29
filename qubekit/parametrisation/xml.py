from typing import TYPE_CHECKING, List, Optional

from openmm import System, app
from typing_extensions import Literal

from qubekit.parametrisation.base_parametrisation import Parametrisation

if TYPE_CHECKING:
    from qubekit.molecules import Ligand


class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    type: Literal["XML"] = "XML"

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def _improper_torsion_ordering(cls) -> str:
        return "default"

    def _build_system(
        self, molecule: "Ligand", input_files: Optional[List[str]] = None
    ) -> System:
        """Serialise the input XML system using openmm."""

        modeller = app.Modeller(
            molecule.to_openmm_topology(), molecule.openmm_coordinates()
        )
        xml = None
        if input_files is not None:
            for file in input_files:
                if file.endswith(".xml"):
                    xml = file
                    break
        # if we did not find one guess the name
        xml = xml or f"{molecule.name}.xml"

        forcefield = app.ForceField(xml)
        # Check for virtual sites
        try:
            system = forcefield.createSystem(
                modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None
            )
        except ValueError:
            print("Virtual sites were found in the xml file")
            modeller.addExtraParticles(forcefield)
            system = forcefield.createSystem(
                modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None
            )

        return system
