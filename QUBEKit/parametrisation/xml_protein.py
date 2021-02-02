#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.ligand import Protein
from QUBEKit.proteins.protein_tools import qube_general

from collections import OrderedDict
from copy import deepcopy
from typing import Optional, List

from simtk.openmm import System, app
import xml.etree.ElementTree as ET


class XMLProtein(Parametrisation):
    """Read in the parameters for a proteins from the QUBEKit_general XML file and store them into the proteins."""

    def __init__(self, fftype='QUBE_general_pi.xml'):
        super().__init__(fftype)

    def build_system(self, protein: Protein, input_files: Optional[List[str]] = None) -> System:
        """Serialise the input XML system using openmm."""

        # TODO add support for other protein forcefields
        # write out the qube forcefield
        if self.fftype == "QUBE_general_pi.xml":
            qube_general()
            pdb = app.PDBFile(f'{protein.name}.pdb')
            modeller = app.Modeller(pdb.topology, pdb.positions)

            forcefield = app.ForceField(self.fftype)

            system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

            return system
        else:
            raise NotImplementedError()

    def gather_parameters(self, protein: Protein) -> Protein:
        """
        This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for atom in protein.atoms:
            protein.AtomTypes[atom.atom_index] = [atom.atom_name, f'QUBE_{atom.atom_index}', atom.atom_name]

        input_xml_file = 'serialised.xml'
        in_root = ET.parse(input_xml_file).getroot()

        # Extract all bond data
        for Bond in in_root.iter('Bond'):
            protein.HarmonicBondForce[(int(Bond.get('p1')), int(Bond.get('p2')))] = [Bond.get('d'), Bond.get('k')]

        # before we continue update the protein class
        protein.update()

        # Extract all angle data
        for Angle in in_root.iter('Angle'):
            protein.HarmonicAngleForce[int(Angle.get('p1')), int(Angle.get('p2')), int(Angle.get('p3'))] = [
                Angle.get('a'), Angle.get('k')]

        # Extract all non-bonded data
        i = 0
        for Atom in in_root.iter('Particle'):
            if "eps" in Atom.attrib:
                protein.NonbondedForce[i] = [float(Atom.get('q')), float(Atom.get('sig')), float(Atom.get('eps'))]
                protein.atoms[i].partial_charge = float(Atom.get('q'))
                i += 1

        # Extract all of the torsion data
        phases = ['0', '3.141592653589793', '0', '3.141592653589793']
        for Torsion in in_root.iter('Torsion'):
            tor_string_forward = tuple(int(Torsion.get(f'p{i}')) for i in range(1, 5))
            tor_string_back = tuple(reversed(tor_string_forward))

            if tor_string_forward not in protein.PeriodicTorsionForce and tor_string_back not in protein.PeriodicTorsionForce:
                protein.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]
            elif tor_string_forward in protein.PeriodicTorsionForce:
                protein.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])
            elif tor_string_back in protein.PeriodicTorsionForce:
                protein.PeriodicTorsionForce[tor_string_back].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])

        # Now we have all of the torsions from the OpenMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        for tor_list in protein.dihedrals.values():
            for torsion in tor_list:
                # change the indexing to check if they match
                if torsion not in protein.PeriodicTorsionForce and tuple(
                        reversed(torsion)) not in protein.PeriodicTorsionForce:
                    protein.PeriodicTorsionForce[torsion] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
                                                                   ['3', '0', '0'], ['4', '0', '3.141592653589793']]

        # Now we need to fill in all blank phases of the Torsions
        for key, val in protein.PeriodicTorsionForce.items():
            vns = ['1', '2', '3', '4']
            if len(val) < 4:
                # now need to add the missing terms from the torsion force
                for force in val:
                    vns.remove(force[0])
                for i in vns:
                    val.append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for force in protein.PeriodicTorsionForce.values():
            force.sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = OrderedDict()
        for improper in protein.improper_torsions:
            for key, val in protein.PeriodicTorsionForce.items():
                # for each improper find the corresponding torsion parameters and save
                if sorted(key) == sorted(improper):
                    # if they match tag the dihedral
                    protein.PeriodicTorsionForce[key].append('Improper')
                    # replace the key with the strict improper order first atom is center
                    improper_torsions[improper] = val

        torsions = deepcopy(protein.PeriodicTorsionForce)
        # now we should remake the torsion store in the ligand
        protein.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object
        for key, val in improper_torsions.items():
            protein.PeriodicTorsionForce[key] = val

        return protein
