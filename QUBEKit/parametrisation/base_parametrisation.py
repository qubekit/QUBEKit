import abc
import xml.etree.ElementTree as ET
from typing import List, Optional

from simtk.openmm import System, XmlSerializer
from typing_extensions import Literal

from QUBEKit.forcefield import VirtualSite3Point
from QUBEKit.molecules.ligand import Ligand
from QUBEKit.utils import constants
from QUBEKit.utils.datastructures import SchemaBase
from QUBEKit.utils.exceptions import TopologyMismatch
from QUBEKit.utils.helpers import check_improper_torsion, check_proper_torsion


class Parametrisation(SchemaBase, abc.ABC):
    """
    Class of methods which perform the initial parametrisation for the molecule.
    The Parameters will be stored into the molecule as force group objects which hold individual parameters.

    Important:
        To make sure all parameters are extracted in a consistent manner between engines, all parameters are extracted from a seralised openmm system.
    """

    type: Literal["base"] = "base"

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

    def parametrsie_molecule(
        self, molecule: Ligand, input_files: Optional[List[str]] = None
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
                site_data = dict(
                    p1=float(virtual_site.get("pos1")),
                    p2=float(virtual_site.get("pos2")),
                    p3=float(virtual_site.get("pos3")),
                    parent_index=int(virtual_site.get("p1")),
                    closest_a_index=int(virtual_site.get("p2")),
                    closest_b_index=int(virtual_site.get("p3")),
                    # TODO add support for four coord sites
                    o_weights=[float(virtual_site.get(f"wo{i}")) for i in range(1, 4)],
                    x_weights=[float(virtual_site.get(f"wx{i}")) for i in range(1, 4)],
                    y_weights=[float(virtual_site.get(f"wy{i}")) for i in range(1, 4)],
                    # fake the charge this will be set later
                    charge=0,
                )
                site = VirtualSite3Point(**site_data)
                sites[i] = site

            # Extract all bond data
            for Bond in in_root.iter("Bond"):
                bond = (int(Bond.get("p1")), int(Bond.get("p2")))
                molecule.BondForce.set_parameter(
                    atoms=bond, length=float(Bond.get("d")), k=float(Bond.get("k"))
                )

            # Extract all angle data
            for Angle in in_root.iter("Angle"):
                angle = int(Angle.get("p1")), int(Angle.get("p2")), int(Angle.get("p3"))
                molecule.AngleForce.set_parameter(
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
                        molecule.NonbondedForce.set_parameter(
                            atoms=(atom_num,),
                            charge=float(Atom.get("q")),
                            sigma=float(Atom.get("sig")),
                            epsilon=float(Atom.get("eps")),
                        )
                        atom_num += 1

            # Check if we found any sites
            if sites:
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
                    molecule.TorsionForce.set_parameter(atoms=tor_str, **data)
                else:
                    try:
                        improper = check_improper_torsion(
                            improper=tor_str, molecule=molecule
                        )
                        molecule.ImproperTorsionForce.set_parameter(
                            atoms=improper, **data
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
