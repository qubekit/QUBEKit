import abc
import xml.etree.ElementTree as ET
from typing import List, Optional

from simtk.openmm import System, XmlSerializer
from typing_extensions import Literal

from qubekit.forcefield import VirtualSite3Point, VirtualSite4Point
from qubekit.molecules.ligand import Ligand
from qubekit.utils import constants
from qubekit.utils.datastructures import StageBase
from qubekit.utils.exceptions import MissingParameterError, TopologyMismatch
from qubekit.utils.helpers import check_improper_torsion, check_proper_torsion


class Parametrisation(StageBase, abc.ABC):
    """
    Class of methods which perform the initial parametrisation for the molecule.
    The Parameters will be stored into the molecule as force group objects which hold individual parameters.

    Important:
        To make sure all parameters are extracted in a consistent manner between engines, all parameters are extracted from a seralised openmm system.
    """

    type: Literal["base"] = "base"

    def start_message(self, **kwargs) -> str:
        return f"Parametrising molecule using local input files."

    def finish_message(self, **kwargs) -> str:
        return "Molecule parameterised and values stored."

    @abc.abstractmethod
    def _build_system(
        self, molecule: Ligand, input_files: Optional[List[str]] = None
    ) -> System:
        """Build a parameterised OpenMM system using the molecule."""
        ...

    @classmethod
    @abc.abstractmethod
    def _improper_torsion_ordering(cls) -> str:
        """Return the improper torsion ordering this parametrisation method uses."""
        ...

    def run(
        self, molecule: Ligand, input_files: Optional[List[str]] = None, **kwargs
    ) -> Ligand:
        """
        Parametrise the ligand using the current engine.

        Args:
            molecule: The qubekit.molecules.Ligand object to be parameterised.
            input_files: The list of input files that should be used to parameterise the molecule.

        Returns:
            A parameterised molecule.
        """
        system = self._build_system(molecule=molecule, input_files=input_files)
        self._serialise_system(system=system)
        self._gather_parameters(molecule=molecule)
        molecule.TorsionForce.ordering = self._improper_torsion_ordering()
        return molecule

    def _serialise_system(self, system: System) -> None:
        """
        Serialise a openMM system to file so that the parameters can be gathered.

        Args:
            system: A parameterised OpenMM system.
        """
        xml = XmlSerializer.serializeSystem(system)
        with open("serialised.xml", "w+") as out:
            out.write(xml)

    def _gather_parameters(self, molecule: Ligand) -> Ligand:
        """
        This method parses the serialised xml file and collects the parameters and
        stores them back into the correct force group in the ligand.

        Args:
            molecule: The molecule to be parameterised.

        Returns:
            A parameterised molecule.
        """
        # store sites under the site index in the system
        sites = {}

        try:
            in_root = ET.parse("serialised.xml").getroot()

            # Extract any virtual site data only supports local coords atm, charges are added later
            for i, virtual_site in enumerate(in_root.iter("LocalCoordinatesSite")):
                limit = 4 if virtual_site.get("wx4") is None else 5
                site_data = dict(
                    p1=float(virtual_site.get("pos1")),
                    p2=float(virtual_site.get("pos2")),
                    p3=float(virtual_site.get("pos3")),
                    parent_index=int(virtual_site.get("p1")),
                    closest_a_index=int(virtual_site.get("p2")),
                    closest_b_index=int(virtual_site.get("p3")),
                    o_weights=[
                        float(virtual_site.get(f"wo{j}")) for j in range(1, limit)
                    ],
                    x_weights=[
                        float(virtual_site.get(f"wx{j}")) for j in range(1, limit)
                    ],
                    y_weights=[
                        float(virtual_site.get(f"wy{j}")) for j in range(1, limit)
                    ],
                    # fake the charge, this will be set later
                    charge=0,
                )
                if virtual_site.get("wx4") is None:
                    site = VirtualSite3Point(**site_data)
                else:
                    site_data["closest_c_index"] = int(virtual_site.get("p4"))
                    site = VirtualSite4Point(**site_data)
                sites[i] = site

            # Extract all bond data
            for Bond in in_root.iter("Bond"):
                bond = (int(Bond.get("p1")), int(Bond.get("p2")))
                molecule.BondForce.create_parameter(
                    atoms=bond, length=float(Bond.get("d")), k=float(Bond.get("k"))
                )

            # Extract all angle data
            for Angle in in_root.iter("Angle"):
                angle = int(Angle.get("p1")), int(Angle.get("p2")), int(Angle.get("p3"))
                molecule.AngleForce.create_parameter(
                    atoms=angle, angle=float(Angle.get("a")), k=float(Angle.get("k"))
                )

            # Extract all non-bonded data, do not add virtual site info to the nonbonded list
            atom_num, site_num = 0, 0
            for Atom in in_root.iter("Particle"):
                if "q" in Atom.attrib:
                    if atom_num >= molecule.n_atoms:
                        sites[site_num].charge = float(Atom.get("q"))
                        site_num += 1
                    else:
                        molecule.NonbondedForce.create_parameter(
                            atoms=(atom_num,),
                            charge=float(Atom.get("q")),
                            sigma=float(Atom.get("sig")),
                            epsilon=float(Atom.get("eps")),
                        )
                        atom_num += 1

            # Check if we found any sites
            if sites:
                # Ensure old sites are removed
                molecule.extra_sites.clear_sites()

                for site in sites.values():
                    molecule.extra_sites.add_site(site=site)

            # Extract all of the torsion data
            for Torsion in in_root.iter("Torsion"):
                k = float(Torsion.get("k"))
                # if k=0 there is no value in saving
                if k == 0:
                    continue
                tor_str = tuple(int(Torsion.get(f"p{i}")) for i in range(1, 5))
                phase = float(Torsion.get("phase"))
                if phase == 3.141594:
                    phase = constants.PI
                p = Torsion.get("periodicity")
                data = {"k" + p: k, "periodicity" + p: int(p), "phase" + p: phase}
                # check if the torsion is proper or improper
                if check_proper_torsion(torsion=tor_str, molecule=molecule):
                    try:
                        torsion = molecule.TorsionForce[tor_str]
                        torsion.update(**data)
                    except MissingParameterError:
                        molecule.TorsionForce.create_parameter(atoms=tor_str, **data)
                else:
                    try:
                        improper_str = check_improper_torsion(
                            improper=tor_str, molecule=molecule
                        )
                        try:
                            torsion = molecule.ImproperTorsionForce[improper_str]
                            torsion.update(**data)
                        except MissingParameterError:
                            molecule.ImproperTorsionForce.create_parameter(
                                atoms=improper_str, **data
                            )
                    except TopologyMismatch:
                        raise RuntimeError(
                            f"Found a torsion that is not proper or improper {tor_str}"
                        )

            return molecule
        except FileNotFoundError:
            # Check what parameter engine we are using if not none then raise an error
            if molecule.parameter_engine != "none":
                raise FileNotFoundError("Molecule could not be serialised from OpenMM")
