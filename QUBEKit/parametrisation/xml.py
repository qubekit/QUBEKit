#!/usr/bin/env python3

from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.parametrisation.base_parametrisation import Parametrisation

import os

import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from simtk.openmm import app, XmlSerializer


@for_all_methods(timer_logger)
class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS'):

        super().__init__(molecule, input_file, fftype)

        # self.check_xml()
        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'XML input ' + self.fftype
        self.molecule.combination = 'opls'

    def check_xml(self):
        """
        This function will check through the xml provided
        and ensure that the connections between the mol2/pdb match the xml this is only for boron compounds
        :return: edited xml file
        """

        re_write = False

        # First parse the xml file
        in_root = ET.parse(self.molecule.name + '.xml').getroot()

        # Now record all of the bonds in the file
        bonds = []
        for Bond in in_root.iter('Bond'):
            # There are two Bond entries and we want the first type
            try:
                bonds.append((int(Bond.get('from')), int(Bond.get('to'))))
            except TypeError:
                break

        # Now make sure the amount of bonds match
        if self.molecule.bond_lengths is not None and len(self.molecule.bond_lengths) != len(bonds):
            for bond in self.molecule.bond_lengths:
                zeroed_bond = (bond[0] - 1, bond[1] - 1)
                if zeroed_bond not in bonds and tuple(reversed(zeroed_bond)) not in bonds:
                    print(f'Warning parameters missing for bond {self.molecule.atom_names[zeroed_bond[0]]}-'
                          f'{self.molecule.atom_names[zeroed_bond[1]]}, adding estimate')

                    re_write = True

                    # Now add the bond tag and parameter tags
                    bond_root = in_root.find('Residues/Residue')
                    ET.SubElement(bond_root, 'Bond', attrib={'from': str(zeroed_bond[0]), 'to': str(zeroed_bond[1])})

                    # Now add the general parameters these will be replaced by the seminario method anyway
                    param_root = in_root.find('HarmonicBondForce')
                    ET.SubElement(param_root, 'Bond', attrib={'class1': f'{self.molecule.coords["input"][zeroed_bond[0]][0]}{800 + zeroed_bond[0]}',
                                                              'class2': f'{self.molecule.coords["input"][zeroed_bond[1]][0]}{800 + zeroed_bond[1]}',
                                                              'length': str(0.140000), 'k': str(392459.200000)})

        # Record all of the angle parameters
        angles = []
        for Angle in in_root.iter('Angle'):
            angles.append((int(Angle.get('class1')[-2:]), int(Angle.get('class2')[-2:]), int(Angle.get('class3')[-2:])))

        # Now we add an angle parameter if it is missing
        if self.molecule.angles is not None and len(self.molecule.angles) != len(angles):
            for angle in self.molecule.angles:
                zeroed_angle = (angle[0] - 1, angle[1] - 1, angle[2] - 1)
                if zeroed_angle not in angles and tuple(reversed(zeroed_angle)) not in angles:
                    print(f'Warning parameters missing for angle {self.molecule.atom_names[zeroed_angle[0]]}-'
                          f'{self.molecule.atom_names[zeroed_angle[1]]}-{self.molecule.atom_names[zeroed_angle[2]]}'
                          f', adding estimate')

                    re_write = True

                    # Now add the general angle parameters
                    angle_root = in_root.find('HarmonicAngleForce')
                    ET.SubElement(angle_root, 'Angle', attrib={'class1': f'{self.molecule.coords["input"][zeroed_angle[0]][0]}{800 + zeroed_angle[0]}',
                                                               'class2': f'{self.molecule.coords["input"][zeroed_angle[1]][0]}{800 + zeroed_angle[1]}',
                                                               'class3': f'{self.molecule.coords["input"][zeroed_angle[2]][0]}{800 + zeroed_angle[2]}',
                                                               'angle': str(2.094395), 'k': str(527.184000)})

        # No dihedrals added as they are added during reading the serialised system

        if re_write:
            # Now we need to remove the old XML and write the new one!
            print('Rewriting the xml with missing parameters')
            os.remove(f'{self.molecule.name}.xml')
            messy = ET.tostring(in_root, 'utf-8')
            pretty_xml_string = parseString(messy).toprettyxml(indent="")
            with open(f'{self.molecule.name}.xml', 'w+') as xml:
                xml.write(pretty_xml_string)

    def serialise_system(self):
        """Serialise the input XML system using openmm."""

        pdb = app.PDBFile(f'{self.molecule.name}.pdb')
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
