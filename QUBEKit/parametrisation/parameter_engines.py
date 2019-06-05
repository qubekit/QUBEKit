#!/usr/bin/env python

from QUBEKit.decorators import for_all_methods, timer_logger

from collections import OrderedDict
from copy import deepcopy
import os
from shutil import copy
from subprocess import run as sub_run
from tempfile import TemporaryDirectory

import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
from simtk.openmm import app, XmlSerializer


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

    def __init__(self, molecule, input_file=None, fftype=None):

        self.molecule = molecule
        self.input_file = input_file
        self.fftype = fftype
        self.combination = 'amber'

        # could be a problem for boron compounds
        # TODO Set back to None if there are none
        self.molecule.AtomTypes = {}
        self.molecule.HarmonicBondForce = {}
        self.molecule.HarmonicAngleForce = {}
        self.molecule.NonbondedForce = OrderedDict()
        self.molecule.PeriodicTorsionForce = OrderedDict()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def gather_parameters(self):
        """
        This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for atom in self.molecule.atoms:
            self.molecule.AtomTypes[atom.index] = [atom.name, 'QUBE_' + str(000 + atom.index),
                                                   str(atom.element) + str(000 + atom.index)]

        in_root = ET.parse('serialised.xml').getroot()

        # Extract all bond data
        for Bond in in_root.iter('Bond'):
            bond = (int(Bond.get('p1')), int(Bond.get('p2')))
            self.molecule.HarmonicBondForce[bond] = [Bond.get('d'), Bond.get('k')]

        # Extract all angle data
        for Angle in in_root.iter('Angle'):
            angle = int(Angle.get('p1')), int(Angle.get('p2')), int(Angle.get('p3'))
            self.molecule.HarmonicAngleForce[angle] = [Angle.get('a'), Angle.get('k')]

        # Extract all non-bonded data
        i = 0
        for Atom in in_root.iter('Particle'):
            if "eps" in Atom.attrib:
                self.molecule.NonbondedForce[i] = [Atom.get('q'), Atom.get('sig'), Atom.get('eps')]
                i += 1

        # Extract all of the torsion data
        phases = ['0', '3.141592653589793', '0', '3.141592653589793']
        for Torsion in in_root.iter('Torsion'):
            tor_str_forward = tuple(int(Torsion.get(f'p{i}')) for i in range(1, 5))
            tor_str_back = tuple(reversed(tor_str_forward))

            if tor_str_forward not in self.molecule.PeriodicTorsionForce and tor_str_back not in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_str_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]

            elif tor_str_forward in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_str_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])

            elif tor_str_back in self.molecule.PeriodicTorsionForce:
                self.molecule.PeriodicTorsionForce[tor_str_back].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])

        # Now we have all of the torsions from the openMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        if self.molecule.dihedrals is not None:
            for tor_list in self.molecule.dihedrals.values():
                for torsion in tor_list:
                    if torsion not in self.molecule.PeriodicTorsionForce and tuple(reversed(torsion)) not in self.molecule.PeriodicTorsionForce:
                        self.molecule.PeriodicTorsionForce[torsion] = [['1', '0', '0'], ['2', '0', '3.141592653589793'], ['3', '0', '0'], ['4', '0', '3.141592653589793']]

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
        for key, val in self.molecule.PeriodicTorsionForce.items():
            val.sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = None
        if self.molecule.improper_torsions is not None:
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
        # Remake the torsion store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object

        if improper_torsions is not None:
            for key, val in improper_torsions.items():
                self.molecule.PeriodicTorsionForce[key] = val


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
                    ET.SubElement(param_root, 'Bond', attrib={'class1': f'{self.molecule.molecule["input"][zeroed_bond[0]][0]}{800 + zeroed_bond[0]}',
                                                              'class2': f'{self.molecule.molecule["input"][zeroed_bond[1]][0]}{800 + zeroed_bond[1]}',
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
                    ET.SubElement(angle_root, 'Angle', attrib={'class1': f'{self.molecule.molecule["input"][zeroed_angle[0]][0]}{800 + zeroed_angle[0]}',
                                                               'class2': f'{self.molecule.molecule["input"][zeroed_angle[1]][0]}{800 + zeroed_angle[1]}',
                                                               'class3': f'{self.molecule.molecule["input"][zeroed_angle[2]][0]}{800 + zeroed_angle[2]}',
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
            self.molecule.AtomTypes[atom.index] = [atom.name, 'QUBE_' + str(atom.index), atom.name]

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
                param = tuple(torsion[i] - 1 for i in range(4))
                if param not in self.molecule.PeriodicTorsionForce and tuple(
                        reversed(param)) not in self.molecule.PeriodicTorsionForce:
                    self.molecule.PeriodicTorsionForce[param] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
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
                if sorted(key) == sorted(tuple([x - 1 for x in improper])):
                    # if they match tag the dihedral
                    self.molecule.PeriodicTorsionForce[key].append('Improper')
                    # replace the key with the strict improper order first atom is center
                    improper_torsions[tuple([x - 1 for x in improper])] = val

        torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # now we should remake the torsion store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object
        for key, val in improper_torsions.items():
            self.molecule.PeriodicTorsionForce[key] = val


@for_all_methods(timer_logger)
class AnteChamber(Parametrisation):
    """
    Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    def __init__(self, molecule, input_file=None, fftype='gaff'):

        super().__init__(molecule, input_file, fftype)

        self.antechamber_cmd()
        self.serialise_system()
        self.gather_parameters()
        self.prmtop = None
        self.inpcrd = None
        self.molecule.parameter_engine = 'AnteChamber ' + self.fftype
        self.molecule.combination = self.combination

    def serialise_system(self):
        """Serialise the amber style files into an openmm object."""

        prmtop = app.AmberPrmtopFile(self.prmtop)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))

    def antechamber_cmd(self):
        """Method to run Antechamber, parmchk2 and tleap."""

        # file paths when moving in and out of temp locations
        cwd = os.getcwd()
        mol2 = os.path.abspath(f'{self.molecule.name}.mol2')
        pdb_path = os.path.abspath(f'{self.molecule.name}.pdb')
        frcmod_file = os.path.abspath(f'{self.molecule.name}.frcmod')
        prmtop_file = os.path.abspath(f'{self.molecule.name}.prmtop')
        inpcrd_file = os.path.abspath(f'{self.molecule.name}.inpcrd')
        ant_log = os.path.abspath('Antechamber.log')

        # Call Antechamber
        # Do this in a temp directory as it produces a lot of files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            copy(pdb_path, 'in.pdb')

            # Call Antechamber
            cmd = f'antechamber -i in.pdb -fi pdb -o out.mol2 -fo mol2 -s 2 -at {self.fftype} -c bcc -nc {self.molecule.charge}'

            with open('ante_log.txt', 'w+') as log:
                sub_run(cmd, shell=True, stdout=log, stderr=log)

            # Ensure command worked
            try:
                # Copy the gaff mol2 and antechamber file back
                copy('out.mol2', mol2)
                copy('ante_log.txt', cwd)
            except FileNotFoundError:
                print('Antechamber could not convert this file type is it a valid pdb?')

            os.chdir(cwd)

        # Work in temp directory due to the amount of files made by antechamber
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            copy(mol2, 'out.mol2')

            # Run parmchk
            with open('Antechamber.log', 'a') as log:
                sub_run(f"parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s {self.fftype}",
                        shell=True, stdout=log, stderr=log)

            # Ensure command worked
            if not os.path.exists('out.frcmod'):
                raise FileNotFoundError('out.frcmod not found parmchk2 failed!')

            # Now get the files back from the temp folder
            copy('out.mol2', mol2)
            copy('out.frcmod', frcmod_file)
            copy('Antechamber.log', ant_log)

        # Now we need to run tleap to get the prmtop and inpcrd files
        with TemporaryDirectory() as temp:
            os.chdir(temp)
            copy(mol2, 'in.mol2')
            copy(frcmod_file, 'in.frcmod')
            copy(ant_log, 'Antechamber.log')

            # make tleap command file
            with open('tleap_commands', 'w+') as tleap:
                tleap.write("""source oldff/leaprc.ff99SB
                               source leaprc.gaff
                               LIG = loadmol2 in.mol2
                               check LIG
                               loadamberparams in.frcmod
                               saveamberparm LIG out.prmtop out.inpcrd
                               quit""")

            # Now run tleap
            with open('Antechamber.log', 'a') as log:
                sub_run('tleap -f tleap_commands', shell=True, stdout=log, stderr=log)

            # Check results present
            if not os.path.exists('out.prmtop') or not os.path.exists('out.inpcrd'):
                raise FileNotFoundError('Neither out.prmtop nor out.inpcrd found; tleap failed!')

            copy('Antechamber.log', ant_log)
            copy('out.prmtop', prmtop_file)
            copy('out.inpcrd', inpcrd_file)
            os.chdir(cwd)

        # Now give the file names to parametrisation method
        self.prmtop = f'{self.molecule.name}.prmtop'
        self.inpcrd = f'{self.molecule.name}.inpcrd'


class OPLSServer(Parametrisation):
    """Is it possible to contact the ligpargen server and get the pdb and xml file for a molecule?"""

    pass


class Default:
    pass
