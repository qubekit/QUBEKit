# TODO write the parametrisation classes for each method antechamber input xml, openFF, etc
# all must return the same dic object that can be stored in the molecule and writen to xml format
# maybe Gromacs as well


from QUBEKit.decorators import for_all_methods, timer_logger


class Parametrisation:
    """Class of functions which perform the initial parametrisation for the molecule.
    The Parameters will be stored into the molecule as dictionaries as this is easy to manipulate and convert
    to a parameter tree.

    Note all parameters gathered here are indexed from 0,
    whereas the ligand object index starts from 1 for all networkx related properties such as bonds!


    Parameters
    ---------
    molecule : QUBEKit molecule object

    input_file : an OpenMM style xml file associated with the molecule object

    fftype : the FF type the molecule will be parametrised with
             only needed in the case of gaff or gaff2 else will be assigned based on class used.

    Returns
    -------
    AtomTypes : dictionary of the atom names, the associated opls type and class type stored under number.

    Residues : dictionary of residue names indexed by the order they appear.

    HarmonicBondForce: dictionary of equilibrium distances and force constants stored under the bond tuple.

    HarmonicAngleForce: dictionary of equilibrium  angles and force constant stored under the angle tuple.

    PeriodicTorsionForce : dictionary of periodicity, barrier and phase stored under the torsion tuple.

    NonbondedForce : dictionary of charge, sigma and epsilon stored under the original atom ordering.
    """

    def __init__(self, molecule, config_file=None, input_file=None, fftype=None):
        self.molecule = molecule
        self.input = input_file
        self.fftype = fftype


@for_all_methods(timer_logger)
class XML(Parametrisation):
    """Read in the parameters for a molecule from an XML file and store them into the molecule."""

    def __init__(self, molecule, config_file=None, input_file=None, fftype='opls'):

        super().__init__(molecule, config_file, input_file, fftype)
        self.parameterise()
        self.molecule.parameter_engine = 'XML input ' + self.fftype

    def read_XML(self):
        """This function reads the input file and saves the XML tree to the molecule."""
        import xml.etree.ElementTree as ET
        tree = ET.parse(self.input)
        root = tree.getroot()
        for child in root:
            # print out the amount of children in each element
            if child.tag == 'PeriodicTorsionForce':
                count = sum(1 for improper in child.iter('Improper'))
                print('impropers= {} normal= {}'.format(count, len(list(child)) - count))
        self.molecule.parameters = tree

    def parameterise(self):
        """This function stores the information into the dict objects."""
        if 'xml' in self.input:
            self.read_XML()
        else:
            raise FileExistsError('No .xml type file found did you supply one?')


@for_all_methods(timer_logger)
class AnteChamber(Parametrisation):
    """Use AnteChamber to parametrise the Ligand first using gaff  or gaff2
    then build and export the xml tree object."""

    def __init__(self, molecule, config_file=None, input_file=None, fftype='gaff'):
        super().__init__(molecule, config_file, input_file, fftype)
        self.parameterise()
        self.molecule.parameter_engine = 'AnteChamber ' + self.fftype

    def parameterise(self):
        """This is the master function of the class
        1 parametrise with Antechamber using gaff or gaff2
        2 convert the parameters to a xml tree object and export to the molecule."""

        # TODO map old atom names to new pdb order
        # then use the frcmod file to find the parameters corresponding to each energy term.
        self.antchamber_cmd()
        self.gather_parameters()

    def gather_parameters(self):
        """Read the frcmod and NEWPDB files then create the XML tree."""
        pass

    def antchamber_cmd(self):
        """Method to run Antechamber based on the given commands."""
        print(self.fftype)


@for_all_methods(timer_logger)
class OpenFF(Parametrisation):
    """This class uses the openFF in openeye to parametrise the molecule using frost.
    A serialized XML is then reformatted into a standard XML tree and saved."""

    def __init__(self, molecule, config_file=None, input_file=None, fftype='frost'):

        super().__init__(molecule, config_file, input_file, fftype)
        self.parametrise()
        self.molecule.parameter_engine = 'OpenFF ' + self.fftype

    def parametrise(self):
        """This is the master function of the class
        1 parametrise the molecule with frost and serialize the system into an xml
        2 parse the object and construct the parameter dictionaries
        3 return the parameters to the molecule."""
        self.serialize_system()

        self.gather_parameters()

    def serialize_system(self):
        """Create the OpenMM system parametrise using frost and serialize the system."""

        # Import OpenMM tools
        from simtk import openmm

        # Import the SMIRNOFF forcefield engine and some useful tools
        from openforcefield.typing.engines.smirnoff import ForceField
        from openforcefield.utils import get_data_filename, generateTopologyFromOEMol

        # Import the OpenEye toolkit
        from openeye import oechem

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

        # Serialize the OpenMM system into the xml file
        xml = openmm.XmlSerializer.serializeSystem(system)
        with open('serilized.xml', 'w+') as out:
            out.write(xml)

    def gather_parameters(self):
        """This method parses the serialized xml file and collects the parameters ready to pass them
        to build tree."""

        import xml.etree.ElementTree as ET
        from math import pi

        # Try to gather the AtomTypes first
        for i, atom in enumerate(self.molecule.atom_names):
            self.molecule.AtomTypes[i] = [atom, 'opls_' + str(800 + i), str(self.molecule.molecule[i][0]) + str(800 + i)]

        # Now parse the xml file for the rest of the data
        inputXML_file = 'serilized.xml'
        inXML = ET.parse(inputXML_file)
        in_root = inXML.getroot()

        # Extract all bond data
        for Bond in in_root.iter('Bond'):
            self.molecule.HarmonicBondForce[(int(Bond.get('p1')), int(Bond.get('p2')))] = [Bond.get('d'),
                                                                                  Bond.get('k')]

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
            tor_string_forward = (int(Torsion.get('p1')), int(Torsion.get('p2')), int(Torsion.get(
                'p3')), int(Torsion.get('p4')))
            tor_string_back = (int(Torsion.get('p4')), int(Torsion.get('p3')), int(Torsion.get('p2')), int(Torsion.get(
                'p1')))
            if tor_string_forward not in self.molecule.PeriodicTorsionForce.keys() and tor_string_back not in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward] = [
                    [Torsion.get('periodicity'), Torsion.get('k'), Torsion.get('phase')]]
            elif tor_string_forward in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_forward].append(
                    [Torsion.get('periodicity'), Torsion.get('k'), Torsion.get('phase')])
            elif tor_string_back in self.molecule.PeriodicTorsionForce.keys():
                self.molecule.PeriodicTorsionForce[tor_string_back].append([Torsion.get('periodicity'), Torsion.get('k'), Torsion.get('phase')])

        # Now we need to fill in all blank phases of the Torsions
        for key in self.molecule.PeriodicTorsionForce.keys():
            Vns = ['1', '2', '3', '4']
            if len(self.molecule.PeriodicTorsionForce[key]) < 4:
                # now need to add the missing terms from the torsion force
                for i in range(len(self.molecule.PeriodicTorsionForce[key])):
                    Vns.remove(self.molecule.PeriodicTorsionForce[key][i][0])
                for i in Vns:
                    self.molecule.PeriodicTorsionForce[key].append([i, '0', phases[int(i) - 1]])
        # sort by periodicity using lambda function
        for key in self.molecule.PeriodicTorsionForce.keys():
            self.molecule.PeriodicTorsionForce[key].sort(key=lambda x: x[0])


@for_all_methods(timer_logger)
class BOSS(Parametrisation):
    pass
