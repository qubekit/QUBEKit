import xml.etree.ElementTree as ET
from collections import OrderedDict
from copy import deepcopy

from QUBEKit.forcefield import VirtualSite3Point
from QUBEKit.molecules.ligand import Ligand
from QUBEKit.utils import constants
from QUBEKit.utils.helpers import check_improper_torsion, check_proper_torsion


class Parametrisation:
    """
    Class of methods which perform the initial parametrisation for the molecule.
    The Parameters will be stored into the molecule as dictionaries as this is easy to manipulate and convert
    to a parameter tree.

    Note all parameters gathered here are indexed from 0,
    whereas the ligand object indices start from 1 for all networkx related properties such as bonds!


    Parameters
    ---------
    molecule : QUBEKit molecule object

    input_file : an OpenMM style xml file associated with the molecule object

    fftype : the FF type the molecule will be parametrised with
             only needed in the case of gaff or gaff2 else will be assigned based on class used.

    Returns
    -------
    AtomTypes : dictionary of the atom names, the associated OPLS type and class type stored under number.
                {0: [C00, OPLS_800, C800]}

    Residues : dictionary of residue names indexed by the order they appear.

    HarmonicBondForce : dictionary of equilibrium distances and force constants stored under the bond tuple.
                        {(0, 1): [eqr=456, fc=984375]}

    HarmonicAngleForce : dictionary of equilibrium  angles and force constant stored under the angle tuple.

    PeriodicTorsionForce : dictionary of periodicity, barrier and phase stored under the torsion tuple.

    NonbondedForce : dictionary of charge, sigma and epsilon stored under the original atom ordering.
    """

    def __init__(self, molecule: Ligand, input_file=None, fftype=None):

        self.molecule = molecule
        self.input_file = input_file
        self.fftype = fftype

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__!r})"

    def gather_parameters(self):
        """
        This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.

        # TODO would conversion be better from the openmm system object?
        # This way we can query if a particle is an extra site, current method breaks if sites are not last in the system.?
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
                self.molecule.BondForce.set_parameter(
                    atoms=bond, length=float(Bond.get("d")), k=float(Bond.get("k"))
                )

            # Extract all angle data
            for Angle in in_root.iter("Angle"):
                angle = int(Angle.get("p1")), int(Angle.get("p2")), int(Angle.get("p3"))
                self.molecule.AngleForce.set_parameter(
                    atoms=angle, angle=float(Angle.get("a")), k=float(Angle.get("k"))
                )

            # Extract all non-bonded data, do not add virtual site info to the nonbonded list
            atom_num, site_num = 0, 0
            for Atom in in_root.iter("Particle"):
                if "q" in Atom.attrib:
                    if atom_num >= self.molecule.n_atoms:
                        sites[site_num].charge = float(Atom.get("q"))
                        site_num += 1
                    else:
                        self.molecule.NonbondedForce.set_parameter(
                            atoms=(atom_num,),
                            charge=float(Atom.get("q")),
                            sigma=float(Atom.get("sig")),
                            epsilon=float(Atom.get("eps")),
                        )
                        atom_num += 1

            # Check if we found any sites
            if sites:
                for site in sites.values():
                    self.molecule.extra_sites.add_site(site=site)

            # Extract all of the torsion data
            for Torsion in in_root.iter("Torsion"):
                tor_str = tuple(int(Torsion.get(f"p{i}")) for i in range(1, 5))
                # check if the torsion is proper or improper
                # grab only k and p, we use a fixed p-phase relation.
                k = float(Torsion.get("k"))
                data = {"k" + Torsion.get("periodicity"): k}
                if check_proper_torsion(torsion=tor_str, molecule=self.molecule):
                    self.molecule.TorsionForce.set_parameter(atoms=tor_str, **data)
                elif check_improper_torsion(improper=tor_str, molecule=self.molecule):
                    self.molecule.ImproperTorsionForce.set_parameter(
                        atoms=tor_str, **data
                    )
                else:
                    raise RuntimeError(
                        f"Found a torsion that is not proper or improper {tor_str}"
                    )

        except FileNotFoundError:
            # Check what parameter engine we are using if not none then raise an error
            if self.molecule.parameter_engine != "none":
                raise FileNotFoundError("Molecule could not be serialised from OpenMM")
        # # Now we have all of the torsions from the OpenMM system
        # # we should check if any torsions we found in the molecule do not have parameters
        # # if they don't, give them the default 0 parameter this will not change the energy
        # if self.molecule.dihedrals is not None:
        #     for tor_list in self.molecule.dihedrals.values():
        #         for torsion in tor_list:
        #             if (
        #                 torsion not in self.molecule.PeriodicTorsionForce
        #                 and tuple(reversed(torsion))
        #                 not in self.molecule.PeriodicTorsionForce
        #             ):
        #                 self.molecule.PeriodicTorsionForce[torsion] = [
        #                     [1, 0, 0],
        #                     [2, 0, constants.PI],
        #                     [3, 0, 0],
        #                     [4, 0, constants.PI],
        #                 ]
        # torsions = [sorted(key) for key in self.molecule.PeriodicTorsionForce.keys()]
        # if self.molecule.improper_torsions is not None:
        #     for torsion in self.molecule.improper_torsions:
        #         if sorted(torsion) not in torsions:
        #             # The improper torsion is missing and should be added with no energy
        #             self.molecule.PeriodicTorsionForce[torsion] = [
        #                 [1, 0, 0],
        #                 [2, 0, constants.PI],
        #                 [3, 0, 0],
        #                 [4, 0, constants.PI],
        #             ]
        #
        # # Now we need to fill in all blank phases of the Torsions
        # for key, val in self.molecule.PeriodicTorsionForce.items():
        #     vns = [1, 2, 3, 4]
        #     if len(val) < 4:
        #         # now need to add the missing terms from the torsion force
        #         for force in val:
        #             vns.remove(force[0])
        #         for i in vns:
        #             val.append([i, 0, phases[int(i) - 1]])
        # # sort by periodicity using lambda function
        # for val in self.molecule.PeriodicTorsionForce.values():
        #     val.sort(key=lambda x: x[0])
        #
        # # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        # improper_torsions = None
        # if self.molecule.improper_torsions is not None:
        #     improper_torsions = OrderedDict()
        #     for improper in self.molecule.improper_torsions:
        #         for key, val in self.molecule.PeriodicTorsionForce.items():
        #             # for each improper find the corresponding torsion parameters and save
        #             if sorted(key) == sorted(improper):
        #                 # if they match tag the dihedral
        #                 self.molecule.PeriodicTorsionForce[key].append("Improper")
        #                 # replace the key with the strict improper order first atom is center
        #                 improper_torsions.setdefault(improper, []).append(val)
        #
        #     # If the improper has been split across multiple combinations we need to collapse them to one
        #     for improper, params in improper_torsions.items():
        #         if len(params) != 1:
        #             # Now we have to sum the k values across the same terms
        #             new_params = params[0]
        #             for values in params[1:]:
        #                 for i in range(4):
        #                     new_params[i][1] += values[i][1]
        #             # Store the summed k values
        #             improper_torsions[improper] = new_params
        #         else:
        #             improper_torsions[improper] = params[
        #                 0
        #             ]  # This unpacks the list if we only find one term
        #
        # torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # # Remake the torsion; store in the ligand
        # self.molecule.PeriodicTorsionForce = OrderedDict(
        #     (v, k) for v, k in torsions.items() if k[-1] != "Improper"
        # )
        #
        # # Add the improper at the end of the torsion object
        # if improper_torsions is not None:
        #     for key, val in improper_torsions.items():
        #         self.molecule.PeriodicTorsionForce[key] = val
