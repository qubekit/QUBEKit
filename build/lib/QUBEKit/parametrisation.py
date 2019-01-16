from QUBEKit.decorators import for_all_methods, timer_logger

from math import pi
from tempfile import TemporaryDirectory
from shutil import copy
from os import getcwd, chdir, path
from subprocess import call as sub_call

from xml.etree.ElementTree import parse
from simtk.openmm import app, XmlSerializer
from openeye import oechem

from openforcefield.typing.engines.smirnoff import ForceField
from openforcefield.utils import get_data_filename, generateTopologyFromOEMol


class Parametrisation:
    """Class of methods which perform the initial parametrisation for the molecule.
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

    Residues : dictionary of residue names indexed by the order they appear.

    HarmonicBondForce: dictionary of equilibrium distances and force constants stored under the bond tuple.

    HarmonicAngleForce: dictionary of equilibrium  angles and force constant stored under the angle tuple.

    PeriodicTorsionForce : dictionary of periodicity, barrier and phase stored under the torsion tuple.

    NonbondedForce : dictionary of charge, sigma and epsilon stored under the original atom ordering.
    """

    def __init__(self, molecule, input_file=None, fftype=None):

        self.molecule = molecule
        self.input = input_file
        self.fftype = fftype

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def gather_parameters(self):
        """This method parses the serialized xml file and collects the parameters ready to pass them
        to build tree.
        """

        # Try to gather the AtomTypes first
        for i, atom in enumerate(self.molecule.atom_names):
            self.molecule.AtomTypes[i] = [atom, 'opls_' + str(800 + i), str(self.molecule.molecule[i][0]) + str(800 + i)]

        # Now parse the xml file for the rest of the data
        input_xml_file = 'serialized.xml'
        in_root = parse(input_xml_file).getroot()

        # Extract all bond data
        for Bond in in_root.iter('Bond'):
            self.molecule.HarmonicBondForce[(int(Bond.get('p1')), int(Bond.get('p2')))] = [Bond.get('d'), Bond.get('k')]

        # Extract all angle data
        for Angle in in_root.iter('Angle'):
            self.molecule.HarmonicAngleForce[int(Angle.get('p1')), int(Angle.get('p2')), int(Angle.get('p3'))] = [Angle.get('a'), Angle.get('k')]

        # Extract all nonbonded data
        i = 0
        for Atom in in_root.iter('Particle'):
            if "eps" in Atom.attrib:
                self.molecule.NonbondedForce[i] = [Atom.get('q'), Atom.get('sig'), Atom.get('eps')]
                i += 1

        # Extract all of the torsion data
        phases = ['0', str(pi), '0', str(pi)]
        for Torsion in in_root.iter('Torsion'):
            tor_string_forward = (int(Torsion.get('p1')), int(Torsion.get('p2')), int(Torsion.get('p3')),
                                  int(Torsion.get('p4')))
            tor_string_back = (int(Torsion.get('p4')), int(Torsion.get('p3')), int(Torsion.get('p2')),
                               int(Torsion.get('p1')))
            if tor_string_forward not in self.molecule.PeriodicTorsionForce.keys() and tor_string_back not in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), Torsion.get('phase')]]
            elif tor_string_forward in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), Torsion.get('phase')])
            elif tor_string_back in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_back].append([Torsion.get('periodicity'),
                                                                            Torsion.get('k'), Torsion.get('phase')])

        # Now we need to fill in all blank phases of the Torsions
        for key in self.molecule.PeriodicTorsionForce.keys():
            Vns = ['1', '2', '3', '4']
            if len(self.molecule.PeriodicTorsionForce[key]) < 4:
                # now need to add the missing terms from the torsion force
                for force in self.molecule.PeriodicTorsionForce[key]:
                    Vns.remove(force[0])
                for i in Vns:
                    self.molecule.PeriodicTorsionForce[key].append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for key in self.molecule.PeriodicTorsionForce.keys():
            self.molecule.PeriodicTorsionForce[key].sort(key=lambda x: x[0])


@for_all_methods(timer_logger)
class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS'):

        super().__init__(molecule, input_file, fftype)
        self.parametrise()
        self.molecule.parameter_engine = 'XML input ' + self.fftype

    def serialize_system(self):
        """Serialize the input XML system using openmm."""

        pdb = app.PDBFile(self.molecule.filename)
        modeller = app.Modeller(pdb.topology, pdb.positions)

        if self.input:
            forcefield = app.ForceField(self.input)
        elif not self.input:
            forcefield = app.ForceField(self.molecule.name + '.xml')
        else:
            raise FileNotFoundError('No .xml type file found did you supply one?')

        system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)

        xml = XmlSerializer.serializeSystem(system)
        with open('serialized.xml', 'w+') as out:
            out.write(xml)

    # TODO Remove parametrise methods and place contents into inits?
    def parametrise(self):
        """This is the master function and controls the class.
        1. Serialise the system into a correctly formatted xml file
        2. gather the parameters and store them in the molecule parameter dictionaries.
        """

        self.serialize_system()
        self.gather_parameters()


@for_all_methods(timer_logger)
class AnteChamber(Parametrisation):
    """Use AnteChamber to parametrise the Ligand first using gaff or gaff2
    then build and export the xml tree object.
    """

    def __init__(self, molecule, input_file=None, fftype='gaff'):
        super().__init__(molecule, input_file, fftype)
        self.parametrise()
        self.prmtop = None
        self.inpcrd = None
        self.molecule.parameter_engine = 'AnteChamber ' + self.fftype

    def parametrise(self):
        """This is the master function of the class
        1 parametrise with Antechamber using gaff or gaff2
        2 load molecule into tleap to get the prmtop and inpcrd files used by openMM
        3 serialize the openMM system object
        4 convert the parameters to a xml tree object and export to the molecule.
        """

        self.antechamber_cmd()
        self.serialize_system()
        self.gather_parameters()

    def serialize_system(self):
        """Serialise the amber style files into an openmm object."""

        prmtop = app.AmberPrmtopFile(self.prmtop)
        system = prmtop.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

        with open('serialized.xml', 'w+') as out:
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

        # Work in temp directory due to the amount of files made by antechamber
        with TemporaryDirectory() as temp:
            chdir(temp)
            copy(input_file, 'in.pdb')
            # Call Antechamber
            with open('Antechamber.log', 'w+') as log:
                sub_call(f'antechamber -i {input_file} -fi pdb -o out.mol2 -fo mol2 -s 2 -at {self.fftype} -c bcc',
                         shell=True, stdout=log)
            # Ensure command worked
            if not path.exists('out.mol2'):
                raise FileNotFoundError('out.mol2 not found antechamber failed!')
            # Run parmchk
            with open('Antechamber.log', 'a') as log:
                sub_call(f"parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s {self.fftype}", shell=True, stdout=log)
            # Ensure command worked
            if not path.exists('out.frcmod'):
                raise FileNotFoundError('out.frcmod not found parmchk2 failed!')
            # Now get the files back from the temp folder and close
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
                sub_call('tleap -f tleap_commands', shell=True, stdout=log)
            # check results present
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
    """This class uses the openFF in openeye to parametrise the molecule using frost.
    A serialised XML is then stored in the parameter dictionaries.
    """

    def __init__(self, molecule, input_file=None, fftype='frost'):

        super().__init__(molecule, input_file, fftype)
        self.parametrise()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype

    def parametrise(self):
        """This is the master function of the class
        1 parametrise the molecule with frost and serialize the system into an xml
        2 parse the object and construct the parameter dictionaries
        3 return the parameters to the molecule.
        """

        self.serialize_system()
        self.gather_parameters()

    def serialize_system(self):
        """Create the OpenMM system parametrise using frost and serialise the system."""

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
        xml = XmlSerializer.serializeSystem(system)
        with open('serialized.xml', 'w+') as out:
            out.write(xml)


@for_all_methods(timer_logger)
class BOSS(Parametrisation):
    """This class uses the BOSS software to parametrise a molecule using the CM1A/OPLS FF.
    The parameters are then stored in the parameter dictionaries.
    """

    # TODO make sure order is consistent with PDB.
    def __init__(self, molecule, input_file=None, fftype='CM1A/OPLS'):

        super().__init__(molecule, input_file, fftype)
        self.parametrise()
        self.molecule.parameter_engine = 'BOSS ' + self.fftype

    def parametrise(self):
        """This is the master function of the class
        1 parametrise the molecule with CM1A/OPLS
        2 parse the out file and construct the parameter dictionaries
        3 return the parameters to the molecule.
        """

        self.BOSS_cmd()
        self.gather_parameters()

    def BOSS_cmd(self):
        """This method is used to call the required BOSS scripts.
        1 The zmat file with CM1A charges is first generated for the molecule keeping the same pdb order.
        2 A single point calculation is done.
        """

        pass

    def gather_parameters(self):
        """This method parses the BOSS out file and collects the parameters ready to pass them
        to build tree.
        """

        pass
