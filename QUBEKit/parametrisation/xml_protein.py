#!/usr/bin/env python3

from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils.decorators import for_all_methods, timer_logger

from collections import OrderedDict
from copy import deepcopy

import xml.etree.ElementTree as ET
from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class XMLProtein(Parametrisation):
    """Read in the parameters for a protein from the QUBEKit_general XML file and store them into the protein."""

    def __init__(self, protein, input_file='QUBE_general_pi.xml', fftype='CM1A/OPLS'):

        super().__init__(protein, input_file, fftype)

        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'XML input ' + self.fftype
        self.molecule.combination = 'opls'

    def serialise_system(self):
        """Serialise the input XML system using openmm."""

        pdb = app.PDBFile(self.molecule.filename)
        modeller = app.Modeller(pdb.topology, pdb.positions)

        if self.input_file:
            forcefield = app.ForceField(self.input_file)
        else:
            try:
                forcefield = app.ForceField(self.molecule.name + '.xml')
            except FileNotFoundError:
                raise FileNotFoundError('No .xml type file found.')

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        xml = XmlSerializer.serializeSystem(system)
        with open('serialised.xml', 'w+') as out:
            out.write(xml)

    def gather_parameters(self):
        """This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for atom in self.molecule.atoms:
            self.molecule.AtomTypes[atom.atom_index] = [atom.name, 'QUBE_' + str(atom.atom_index), atom.name]

        input_xml_file = 'serialised.xml'
        in_root = ET.parse(input_xml_file).getroot()

        # Extract all bond data
        for Bond in in_root.iter('Bond'):
            self.molecule.HarmonicBondForce[(int(Bond.get('p1')), int(Bond.get('p2')))] = [Bond.get('d'), Bond.get('k')]

        # before we continue update the protein class
        self.molecule.update()

        # Extract all angle data
        for Angle in in_root.iter('Angle'):
            self.molecule.HarmonicAngleForce[int(Angle.get('p1')), int(Angle.get('p2')), int(Angle.get('p3'))] = [
                Angle.get('a'), Angle.get('k')]

        # Extract all non-bonded data
        i = 0
        for Atom in in_root.iter('Particle'):
            if "eps" in Atom.attrib:
                self.molecule.NonbondedForce[i] = [Atom.get('q'), Atom.get('sig'), Atom.get('eps')]
                i += 1

        # Extract all of the torsion data
        phases = ['0', '3.141592653589793', '0', '3.141592653589793']
        for Torsion in in_root.iter('Torsion'):
            tor_string_forward = tuple(int(Torsion.get(f'p{i}')) for i in range(1, 5))
            tor_string_back = tuple(reversed(tor_string_forward))

            if tor_string_forward not in self.molecule.PeriodicTorsionForce and tor_string_back not in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]
            elif tor_string_forward in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])
            elif tor_string_back in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_string_back].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])
        # Now we have all of the torsions from the openMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        for tor_list in self.molecule.dihedrals.values():
            for torsion in tor_list:
                # change the indexing to check if they match
                if torsion not in self.molecule.PeriodicTorsionForce and tuple(
                        reversed(torsion)) not in self.molecule.PeriodicTorsionForce:
                    self.molecule.PeriodicTorsionForce[torsion] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
                                                                 ['3', '0', '0'], ['4', '0', '3.141592653589793']]

        # Now we need to fill in all blank phases of the Torsions
        for key, val in self.molecule.PeriodicTorsionForce.items():
            vns = ['1', '2', '3', '4']
            if len(val) < 4:
                # now need to add the missing terms from the torsion force
                for force in val:
                    vns.remove(force[0])
                for i in vns:
                    val.append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for force in self.molecule.PeriodicTorsionForce.values():
            force.sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = OrderedDict()
        for improper in self.molecule.improper_torsions:
            for key, val in self.molecule.PeriodicTorsionForce.items():
                # for each improper find the corresponding torsion parameters and save
                if sorted(key) == sorted(improper):
                    # if they match tag the dihedral
                    self.molecule.PeriodicTorsionForce[key].append('Improper')
                    # replace the key with the strict improper order first atom is center
                    improper_torsions[improper] = val

        torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # now we should remake the torsion store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object
        for key, val in improper_torsions.items():
            self.molecule.PeriodicTorsionForce[key] = val
