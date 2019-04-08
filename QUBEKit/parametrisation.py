#!/usr/bin/env python

from QUBEKit.decorators import for_all_methods, timer_logger
from QUBEKit.helpers import append_to_log

from tempfile import TemporaryDirectory
from shutil import copy
from os import getcwd, chdir, path
from subprocess import run as sub_run
from collections import OrderedDict
from copy import deepcopy

from xml.etree.ElementTree import parse as parse_tree
from simtk.openmm import app, XmlSerializer
from openeye import oechem

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import get_data_filename, generateTopologyFromOEMol

# TODO Users should be able to just install ONE of the necessary parametrisation methods and not worry about needing the others too.
#   Is there a nice way of doing this other than try: import <module>; except ImportError: pass ?


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

    HarmonicBondForce: dictionary of equilibrium distances and force constants stored under the bond tuple.
                {(0, 1): [eqr=456, fc=984375]}

    HarmonicAngleForce: dictionary of equilibrium  angles and force constant stored under the angle tuple.

    PeriodicTorsionForce : dictionary of periodicity, barrier and phase stored under the torsion tuple.

    NonbondedForce : dictionary of charge, sigma and epsilon stored under the original atom ordering.
    """

    def __init__(self, molecule, input_file=None, fftype=None, mol2_file=None):

        self.molecule = molecule
        self.input_file = input_file
        self.fftype = fftype
        self.gaff_types = {}

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def gather_parameters(self):
        """
        This method parses the serialised xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for i, atom in enumerate(self.molecule.atom_names):
            self.molecule.AtomTypes[i] = [atom, 'QUBE_' + str(800 + i),
                                          str(self.molecule.molecule['input'][i][0]) + str(800 + i),
                                          self.gaff_types[atom]]

        input_xml_file = 'serialised.xml'
        in_root = parse_tree(input_xml_file).getroot()

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
            tor_string_forward = tuple(int(Torsion.get(f'p{i}')) for i in range(1, 5))
            tor_string_back = tuple(reversed(tor_string_forward))

            if tor_string_forward not in self.molecule.PeriodicTorsionForce.keys() and tor_string_back not in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]
            elif tor_string_forward in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])
            elif tor_string_back in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_back].append([Torsion.get('periodicity'),
                                                                            Torsion.get('k'), phases[
                                                                                int(Torsion.get('periodicity')) - 1]])
        # Now we have all of the torsions from the openMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        for tor_list in self.molecule.dihedrals.values():
            for torsion in tor_list:
                # change the indexing to check if they match
                param = tuple(torsion[i] - 1 for i in range(4))
                if param not in self.molecule.PeriodicTorsionForce.keys() and tuple(reversed(param)) not in self.molecule.PeriodicTorsionForce.keys():
                    self.molecule.PeriodicTorsionForce[param] = [['1', '0', '0'], ['2', '0', '3.141592653589793'], ['3', '0', '0'], ['4', '0', '3.141592653589793']]

        # Now we need to fill in all blank phases of the Torsions
        for key in self.molecule.PeriodicTorsionForce.keys():
            vns = ['1', '2', '3', '4']
            if len(self.molecule.PeriodicTorsionForce[key]) < 4:
                # now need to add the missing terms from the torsion force
                for force in self.molecule.PeriodicTorsionForce[key]:
                    vns.remove(force[0])
                for i in vns:
                    self.molecule.PeriodicTorsionForce[key].append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for key in self.molecule.PeriodicTorsionForce.keys():
            self.molecule.PeriodicTorsionForce[key].sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = OrderedDict()
        for improper in self.molecule.improper_torsions:
            for key in self.molecule.PeriodicTorsionForce:
                # for each improper find the corresponding torsion parameters and save
                if sorted(key) == sorted(tuple([x - 1 for x in improper])):
                    # if they match tag the dihedral
                    self.molecule.PeriodicTorsionForce[key].append('Improper')
                    # replace the key with the strict improper order first atom is center
                    improper_torsions[tuple([x - 1 for x in improper])] = self.molecule.PeriodicTorsionForce[key]

        torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # Remake the torsion store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object
        for key in improper_torsions.keys():
            self.molecule.PeriodicTorsionForce[key] = improper_torsions[key]

    def get_gaff_types(self, fftype='gaff', file=None):
        """Convert the pdb file into a mol2 antechamber file and get the gaff atom types
        and gaff bonds if there were """

        # call Antechamber to convert if we don't have the mol2 file
        if file is None:
            cwd = getcwd()

            # do this in a temp directory as it produces a lot of files
            pdb = path.abspath(self.molecule.filename)
            mol2 = path.abspath(f'{self.molecule.name}.mol2')
            file = mol2

            with TemporaryDirectory() as temp:
                chdir(temp)
                copy(pdb, 'in.pdb')

                # call antechamber
                with open('Antechamber.log', 'w+') as log:
                    sub_run(f'antechamber -i in.pdb -fi pdb -o out.mol2 -fo mol2 -s 2 -at '
                             f'{fftype} -c bcc', shell=True, stdout=log)

                # Ensure command worked
                if not path.exists('out.mol2'):
                    raise FileNotFoundError('out.mol2 not found antechamber failed!')


                # now copy the file back from the folder
                copy('out.mol2', mol2)
                chdir(cwd)

        # Get the gaff atom types and bonds in case we don't have this info
        gaff_bonds = {}
        with open(file, 'r') as mol_in:
            atoms = False
            bonds = False
            for line in mol_in.readlines():

                # TODO Surely this can be simplified?!
                if '@<TRIPOS>ATOM' in line:
                    atoms = True
                    continue
                elif '@<TRIPOS>BOND' in line:
                    atoms = False
                    bonds = True
                    continue
                elif '@<TRIPOS>SUBSTRUCTURE' in line:
                    bonds = False
                    continue
                if atoms:
                    self.gaff_types[self.molecule.atom_names[int(line.split()[0]) - 1]] = str(line.split()[5])
                if bonds:
                    try:
                        gaff_bonds[int(line.split()[1])].append(int(line.split()[2]))
                    except KeyError:
                        gaff_bonds[int(line.split()[1])] = [int(line.split()[2])]

        append_to_log(f'GAFF types: {self.gaff_types}', msg_type='minor')

        # Check if the molecule already has bonds; if not apply these bonds
        if not list(self.molecule.topology.edges):
            # add the bonds to the molecule
            for key, value in gaff_bonds.items():
                for node in value:
                    self.molecule.topology.add_edge(key, node)

            self.molecule.update()

            # Warning this rewrites the pdb file and re
            # Write a new pdb with the connection information
            self.molecule.write_pdb(input_type='input', name=f'{self.molecule.name}_qube')
            self.molecule.filename = f'{self.molecule.name}_qube.pdb'
            print(f'Molecule connections updated new pdb file made and used: {self.molecule.name}_qube.pdb')
            # Update the input file name for the xml
            self.input_file = f'{self.molecule.name}.xml'


@for_all_methods(timer_logger)
class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS', mol2_file=None):

        super().__init__(molecule, input_file, fftype, mol2_file)

        self.get_gaff_types(fftype='gaff', file=mol2_file)
        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'XML input ' + self.fftype

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


@for_all_methods(timer_logger)
class XMLProtein(Parametrisation):
    """Read in the parameters for a protein from the QUBEKit_general XML file and store them into the protein."""

    def __init__(self, protein, input_file='QUBE_general_pi.xml', fftype='CM1A/OPLS'):

        super().__init__(protein, input_file, fftype)

        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'XML input ' + self.fftype

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
        for i, atom in enumerate(self.molecule.atom_names):
            self.molecule.AtomTypes[i] = [atom, 'QUBE_' + str(i),
                                          str(self.molecule.molecule['input'][i][0]) + str(i)]

        input_xml_file = 'serialised.xml'
        in_root = parse_tree(input_xml_file).getroot()

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

            if tor_string_forward not in self.molecule.PeriodicTorsionForce.keys() and tor_string_back not in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]]]
            elif tor_string_forward in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), phases[int(Torsion.get('periodicity')) - 1]])
            elif tor_string_back in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_back].append([Torsion.get('periodicity'),
                                                                            Torsion.get('k'), phases[
                                                                                int(Torsion.get('periodicity')) - 1]])
        # Now we have all of the torsions from the openMM system
        # we should check if any torsions we found in the molecule do not have parameters
        # if they don't give them the default 0 parameter this will not change the energy
        for tor_list in self.molecule.dihedrals.values():
            for torsion in tor_list:
                # change the indexing to check if they match
                param = tuple(torsion[i] - 1 for i in range(4))
                if param not in self.molecule.PeriodicTorsionForce.keys() and tuple(
                        reversed(param)) not in self.molecule.PeriodicTorsionForce.keys():
                    self.molecule.PeriodicTorsionForce[param] = [['1', '0', '0'], ['2', '0', '3.141592653589793'],
                                                                 ['3', '0', '0'], ['4', '0', '3.141592653589793']]

        # Now we need to fill in all blank phases of the Torsions
        for key in self.molecule.PeriodicTorsionForce.keys():
            vns = ['1', '2', '3', '4']
            if len(self.molecule.PeriodicTorsionForce[key]) < 4:
                # now need to add the missing terms from the torsion force
                for force in self.molecule.PeriodicTorsionForce[key]:
                    vns.remove(force[0])
                for i in vns:
                    self.molecule.PeriodicTorsionForce[key].append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for key in self.molecule.PeriodicTorsionForce.keys():
            self.molecule.PeriodicTorsionForce[key].sort(key=lambda x: x[0])

        # now we need to tag the proper and improper torsions and reorder them so the first atom is the central
        improper_torsions = OrderedDict()
        for improper in self.molecule.improper_torsions:
            for key in self.molecule.PeriodicTorsionForce:
                # for each improper find the corresponding torsion parameters and save
                if sorted(key) == sorted(tuple([x - 1 for x in improper])):
                    # if they match tag the dihedral
                    self.molecule.PeriodicTorsionForce[key].append('Improper')
                    # replace the key with the strict improper order first atom is center
                    improper_torsions[tuple([x - 1 for x in improper])] = self.molecule.PeriodicTorsionForce[key]

        torsions = deepcopy(self.molecule.PeriodicTorsionForce)
        # now we should remake the torsion store in the ligand
        self.molecule.PeriodicTorsionForce = OrderedDict((v, k) for v, k in torsions.items() if k[-1] != 'Improper')
        # now we need to add the impropers at the end of the torsion object
        for key in improper_torsions.keys():
            self.molecule.PeriodicTorsionForce[key] = improper_torsions[key]


@for_all_methods(timer_logger)
class AnteChamber(Parametrisation):
    """
    Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    def __init__(self, molecule, input_file=None, fftype='gaff', mol2_file=None):

        super().__init__(molecule, input_file, fftype, mol2_file)

        self.antechamber_cmd()
        self.serialise_system()
        self.gather_parameters()
        self.prmtop = None
        self.inpcrd = None
        self.molecule.parameter_engine = 'AnteChamber ' + self.fftype

    def serialise_system(self):
        """Serialise the amber style files into an openmm object."""

        prmtop = app.AmberPrmtopFile(self.prmtop)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))

    def antechamber_cmd(self):
        """Method to run Antechamber, parmchk2 and tleap."""

        # file paths when moving in and out of temp locations
        cwd = getcwd()
        input_file = path.abspath(self.molecule.filename)
        mol2 = path.abspath(f'{self.molecule.name}.mol2')
        frcmod_file = path.abspath(f'{self.molecule.name}.frcmod')
        prmtop_file = path.abspath(f'{self.molecule.name}.prmtop')
        inpcrd_file = path.abspath(f'{self.molecule.name}.inpcrd')
        ant_log = path.abspath('Antechamber.log')

        # Call Antechamber
        self.get_gaff_types(fftype=self.fftype)

        # Work in temp directory due to the amount of files made by antechamber
        with TemporaryDirectory() as temp:
            chdir(temp)
            copy(mol2, 'out.mol2')


            # Run parmchk
            with open('Antechamber.log', 'a') as log:
                sub_run(f"parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s {self.fftype}", shell=True, stdout=log)

            # Ensure command worked
            if not path.exists('out.frcmod'):
                raise FileNotFoundError('out.frcmod not found parmchk2 failed!')

            # Now get the files back from the temp folder
            copy('out.mol2', mol2)
            copy('out.frcmod', frcmod_file)
            copy('Antechamber.log', ant_log)

        # Now we need to run tleap to get the prmtop and inpcrd files
        with TemporaryDirectory() as temp:
            chdir(temp)
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
                sub_run('tleap -f tleap_commands', shell=True, stdout=log)

            # Check results present
            if not path.exists('out.prmtop') or not path.exists('out.inpcrd'):
                raise FileNotFoundError('Neither out.prmtop nor out.inpcrd found; tleap failed!')

            copy('Antechamber.log', ant_log)
            copy('out.prmtop', prmtop_file)
            copy('out.inpcrd', inpcrd_file)
            chdir(cwd)

        # Now give the file names to parametrisation method
        self.prmtop = f'{self.molecule.name}.prmtop'
        self.inpcrd = f'{self.molecule.name}.inpcrd'


@for_all_methods(timer_logger)
class OpenFF(Parametrisation):
    """
    This class uses the openFF in openeye to parametrise the molecule using frost.
    A serialised XML is then stored in the parameter dictionaries.
    """

    def __init__(self, molecule, input_file=None, fftype='frost', mol2_file=None):
        super().__init__(molecule, input_file, fftype, mol2_file)

        self.get_gaff_types(mol2_file)
        self.serialise_system()
        self.gather_parameters()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype

    def serialise_system(self):
        """Create the OpenMM system; parametrise using frost; serialise the system."""

        # Load molecule using OpenEye tools
        mol = oechem.OEGraphMol()
        ifs = oechem.oemolistream(self.molecule.filename)
        flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
        ifs.SetFlavor(oechem.OEFormat_MOL2, flavor)
        oechem.OEReadMolecule(ifs, mol)
        oechem.OETriposAtomNames(mol)

        # Load a SMIRNOFF small molecule forcefield for alkanes, ethers, and alcohols
        forcefield = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

        # Create the OpenMM system
        topology = generateTopologyFromOEMol(mol)
        system = forcefield.createSystem(topology, [mol])

        # Serialise the OpenMM system into the xml file
        with open('serialised.xml', 'w+') as out:
            out.write(XmlSerializer.serializeSystem(system))

        # get the gaff atom types
        self.get_gaff_types()


@for_all_methods(timer_logger)
class BOSS(Parametrisation):
    """
    This class uses the BOSS software to parametrise a molecule using the CM1A/OPLS FF.
    The parameters are then stored in the parameter dictionaries.
    """

    # TODO make sure order is consistent with PDB.
    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS'):
        super().__init__(molecule, input_file, fftype)

        self.BOSS_cmd()
        self.gather_parameters()
        self.molecule.parameter_engine = 'BOSS ' + self.fftype

    def BOSS_cmd(self):
        """
        This method is used to call the required BOSS scripts.
        1 The zmat file with CM1A charges is first generated for the molecule keeping the same pdb order.
        2 A single point calculation is done.
        """

        pass

    def gather_parameters(self):
        """
        This method parses the BOSS out file and collects the parameters ready to pass them
        to build tree.
        """

        pass
