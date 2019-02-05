#!/usr/bin/env python

from networkx import neighbors, Graph, has_path
from re import sub
from numpy import array, linalg, dot, degrees, cross, arctan2, arccos
from datetime import datetime
from xml.etree.ElementTree import tostring, Element, SubElement, ElementTree
from xml.dom.minidom import parseString
from pickle import dump, load
from collections import OrderedDict


class Ligand:

    def __init__(self, filename, smilesstring=None):

        self.filename = filename
        self.name = filename[:-4]
        self.molecule = None                # List of lists where the inner list is the atom type followed by its coords
        self.topology = None
        self.smiles = smilesstring
        self.angles = None                  # Shows angles based on atom indices (+1) e.g. (1, 2, 4), (1, 2, 5)
        self.dihedrals = None
        self.rotatable = None
        self.scan_order = None
        self.dih_phis = None
        self.bond_lengths = None
        self.angle_values = None
        self.bonds = None
        self.MMoptimized = None
        self.QMoptimized = None
        self.parameter_engine = None
        self.hessian = None
        self.modes = None
        self.atom_names = None
        self.polar = None
        self.xml_tree = None
        self.state = None
        self.QM_scan_energy = {}
        self.MM_scan_energy = {}
        self.descriptors = {}
        self.AtomTypes = {}                 # Basic non-symmetrised atoms types
        self.symmetry_types = []            # Symmetrised atom types
        self.Residues = {}
        self.HarmonicBondForce = {}
        self.HarmonicAngleForce = {}
        self.PeriodicTorsionForce = OrderedDict()
        self.NonbondedForce = OrderedDict()
        self.read_pdb()
        self.find_angles()
        self.find_dihedrals()
        self.find_rotatable_dihedrals()
        self.get_dihedral_values()
        self.get_bond_lengths()
        self.get_angle_values()
        self.log_file = None                # Full log file name used by the run file in special run cases

    element_dict = {'H': 1.008000,      # Group 1
                    'C': 12.011000,     # Group 4
                    'N': 14.007000, 'P': 30.973762,     # Group 5
                    'O': 15.999000, 'S': 32.060000,     # Group 6
                    'F': 18.998403, 'Cl': 35.450000, 'Br': 79.904000, 'I': 126.904470       # Group 7
                    }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def __str__(self, trunc=False):
        """Prints the ligand class objects' names and values one after another with new lines between each.
        Mostly just used for logging and displaying the results at the end of a run.
        If trunc is set to True:
            Check the items being printed:
                If None -> print them as normal
                If they're short (<120 chars) -> print them as normal
                Otherwise -> print a truncated version of them.
        This is called with:   Ligand(filename='').__str__(trunc=True)
        """

        # This is the old __str__ definition which is basically a one-line alternative to the else case below.
        # return '\n'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__))

        return_str = ''
        for item in self.__dict__:
            if trunc:
                # if (None) or (it's smaller than 120 chars): print it as is. Otherwise print a truncated version.
                return_str += f'\n{item} = {self.__dict__[item] if (self.__dict__[item] is None) or (len(str(self.__dict__[item]) + str(item)) < 120) else str(self.__dict__[item])[:121 - len(str(item))] + "..."}'
            else:
                # Return all objects as {ligand object name} = {ligand object value(s)} without any special formatting.
                return_str += f'\n{item} = {self.__dict__[item]}'

        return return_str

    def read_pdb(self, QM=False, MM=False):
        """Reads the input PDB file to find the ATOM or HETATM tags, extracts the elements and xyz coordinates.
        Then reads through the connection tags and builds a connectivity network
        (only works if connections are present in PDB file).
        Bonds are easily found through the edges of the network.
        Can also generate a simple plot of the network.
        """

        with open(self.filename, 'r') as pdb:
            lines = pdb.readlines()

        molecule = []
        self.topology = Graph()
        self.atom_names = []

        # atom counter used for graph node generation
        atom_count = 1
        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                element = str(line[76:78])
                element = sub('[0-9]+', '', element)
                element = element.replace(" ", "")
                self.atom_names.append(str(line.split()[2]))

                # If the element column is missing from the pdb, extract the element from the name.
                if not element:
                    element = str(line.split()[2])[:-1]
                    element = sub('[0-9]+', '', element)

                # Also add the atom number as the node in the graph
                self.topology.add_node(atom_count)
                atom_count += 1
                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54])])

            if 'CONECT' in line:
                # Now look through the connectivity section and add all edges to the graph corresponding to the bonds.
                for i in range(2, len(line.split())):
                    if int(line.split()[i]) != 0:
                        self.topology.add_edge(int(line.split()[1]), int(line.split()[i]))

        # Uncomment the following lines to draw the graph network generated from the pdb.
        # draw(topology, with_labels=True, font_weight='bold')
        # plt.show()

        if QM:
            self.QMoptimized = molecule
        elif MM:
            self.MMoptimized = molecule
        else:
            self.molecule = molecule

        return self

    def find_angles(self):
        """Take the topology graph network and return a list of all angle combinations.
        Checked against OPLS-AA on molecules containing 10-63 angles.
        """

        self.angles = []

        for node in self.topology.nodes:
            bonded = sorted(list(neighbors(self.topology, node)))
            # Check that the atom has more than one bond
            # TODO Reverse this? Currently I don't think it does anything.
            """
            if len(bonded) >= 2:
                raise Exception('')
            """
            if len(bonded) < 2:
                continue

            # Find all possible angle combinations from the list
            for i in range(len(bonded)):
                for j in range(i + 1, len(bonded)):
                    atom1, atom3 = bonded[i], bonded[j]

                    self.angles.append((atom1, node, atom3))

        return self.angles

    def find_dihedrals(self):
        """Take the topology graph network and again return a dictionary of all possible dihedral combinations stored under
        the central bond keys which describe the angle.
        """

        self.dihedrals = {}

        # Work through the network using each edge as a central dihedral bond
        for edge in self.topology.edges:

            for start in list(neighbors(self.topology, edge[0])):

                # Check atom not in main bond
                if start != edge[0] and start != edge[1]:

                    for end in list(neighbors(self.topology, edge[1])):

                        # Check atom not in main bond
                        if end != edge[0] and end != edge[1]:

                            if edge not in self.dihedrals.keys():
                                # Add the central edge as a key the first time it is used
                                self.dihedrals[edge] = [(start, edge[0], edge[1], end)]

                            else:
                                # Add the tuple to the correct key.
                                self.dihedrals[edge].append((start, edge[0], edge[1], end))

        return self.dihedrals

    def find_rotatable_dihedrals(self):
        """For each dihedral in the topology graph network and dihedrals dictionary, work out if the torsion is
        rotatable. Returns a list of dihedral dictionary keys representing the rotatable dihedrals.
        """

        self.rotatable = []

        # For each dihedral key remove the edge from the network
        for key in self.dihedrals.keys():
            self.topology.remove_edge(*key)

            # Check if there is still a path between the two atoms in the edges.
            if not has_path(self.topology, key[0], key[1]):
                self.rotatable.append(key)

            # Add edge back to the network and try next key
            self.topology.add_edge(*key)

        return self.rotatable

    def get_dihedral_values(self, QM=False, MM=False):
        """Taking the molecules' xyz coordinates and dihedrals dictionary, return a dictionary of dihedral
        angle keys and values. Also an option to only supply the keys of the dihedrals you want to calculate.
        """

        self.dih_phis = {}
        # Check if a rotatable tuple list is supplied, else calculate the angles for all dihedrals in the molecule.

        keys = self.rotatable if self.rotatable else list(self.dihedrals.keys())

        if QM:
            molecule = self.QMoptimized
        elif MM:
            molecule = self.MMoptimized
        else:
            molecule = self.molecule

        for key in keys:
            for torsion in self.dihedrals[key]:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1, x2, x3, x4 = [array(molecule[int(torsion[i])-1][1:]) for i in range(4)]
                b1, b2, b3 = x2 - x1, x3 - x2, x4 - x3
                t1 = linalg.norm(b2) * dot(b1, cross(b2, b3))
                t2 = dot(cross(b1, b2), cross(b2, b3))
                self.dih_phis[torsion] = degrees(arctan2(t1, t2))

        return self.dih_phis

    def get_bond_lengths(self, QM=False, MM=False):
        """For the given molecule and topology find the length of all of the bonds."""

        self.bond_lengths = {}

        if QM:
            molecule = self.QMoptimized
        elif MM:
            molecule = self.MMoptimized
        else:
            molecule = self.molecule

        for edge in self.topology.edges:
            atom1 = array(molecule[int(edge[0]) - 1][1:])
            atom2 = array(molecule[int(edge[1]) - 1][1:])
            bond_dist = linalg.norm(atom2 - atom1)
            self.bond_lengths[edge] = bond_dist

        return self.bond_lengths

    def get_angle_values(self, QM=False, MM=False):
        """For the given molecule and list of angle terms measure the angle values,
        then return a dictionary of angles and values.
        """

        self.angle_values = {}

        if QM:
            molecule = self.QMoptimized
        elif MM:
            molecule = self.MMoptimized
        else:
            molecule = self.molecule

        for angle in self.angles:
            x1 = array(molecule[int(angle[0])-1][1:])
            x2 = array(molecule[int(angle[1])-1][1:])
            x3 = array(molecule[int(angle[2])-1][1:])
            b1, b2 = x1 - x2, x3 - x2
            cosine_angle = dot(b1, b2) / (linalg.norm(b1) * linalg.norm(b2))
            self.angle_values[angle] = degrees(arccos(cosine_angle))

        return self.angle_values

    def write_pdb(self, QM=False, MM=False, name=None):
        """Take the current molecule and topology and write a pdb file for the molecule.
        Only for small molecules, not standard residues. No size limit.
        """

        if QM:
            molecule = self.QMoptimized
        elif MM:
            molecule = self.MMoptimized
        else:
            molecule = self.molecule

        with open(f'{name if name else self.name}.pdb', 'w+') as pdb_file:

            # Write out the atomic xyz coordinates
            pdb_file.write(f'REMARK   1 CREATED WITH QUBEKit {datetime.now()}\n')
            pdb_file.write(f'COMPND    {self.name:<20}\n')
            for i, atom in enumerate(molecule):
                pdb_file.write(f'HETATM{i+1:>5}{self.atom_names[i]:>4}  UNL     1{atom[1]:12.3f}{atom[2]:8.3f}{atom[3]:8.3f}  1.00  0.00          {atom[0]:2}\n')

            # Now add the connection terms
            for node in self.topology.nodes:
                bonded = sorted(list(neighbors(self.topology, node)))
                # if len(bonded) > 2:
                pdb_file.write(f'CONECT{node:5}{"".join(f"{x:5}" for x in bonded)}\n')

            pdb_file.write('END\n')

    def write_parameters(self, name=None):
        """Take the molecule's parameter set and write an xml file for the molecule."""

        # First build the xml tree
        self.build_tree()

        tree = self.xml_tree.getroot()
        messy = tostring(tree, 'utf-8')

        pretty_xml_as_string = parseString(messy).toprettyxml(indent="")

        if not name:
            name = self.name

        with open(f'{name}.xml', 'w+') as xml_doc:
            xml_doc.write(pretty_xml_as_string)

    def build_tree(self):
        """Separates the parameters and builds an xml tree ready to be used."""

        # create XML layout
        root = Element('ForceField')
        AtomTypes = SubElement(root, "AtomTypes")
        Residues = SubElement(root, "Residues")
        Residue = SubElement(Residues, "Residue", name="UNK")
        HarmonicBondForce = SubElement(root, "HarmonicBondForce")
        HarmonicAngleForce = SubElement(root, "HarmonicAngleForce")
        PeriodicTorsionForce = SubElement(root, "PeriodicTorsionForce")
        NonbondedForce = SubElement(root, "NonbondedForce", attrib={'coulomb14scale': "0.5", 'lj14scale': "0.5"})

        # Add the AtomTypes
        for i in range(len(self.AtomTypes)):
            SubElement(AtomTypes, "Type", attrib={'name': self.AtomTypes[i][1],
                                                  'class': self.AtomTypes[i][2],
                                                  'element': self.molecule[i][0],
                                                  'mass': str(self.element_dict[self.molecule[i][0]])})
            SubElement(Residue, "Atom", attrib={'name': self.AtomTypes[i][0], 'type': self.AtomTypes[i][1]})

        # add the bonds/connections
        for key in self.HarmonicBondForce.keys():
            SubElement(Residue, "Bond", attrib={'from': str(key[0]), 'to': str(key[1])})
            SubElement(HarmonicBondForce, "Bond", attrib={'class1': self.AtomTypes[key[0]][2],
                                                          'class2': self.AtomTypes[key[1]][2],
                                                          'length': self.HarmonicBondForce[key][0],
                                                          'k': self.HarmonicBondForce[key][1]})

        # add the angles
        for key in self.HarmonicAngleForce.keys():
            SubElement(HarmonicAngleForce, "Angle", attrib={'class1': self.AtomTypes[key[0]][2],
                                                            'class2': self.AtomTypes[key[1]][2],
                                                            'class3': self.AtomTypes[key[2]][2],
                                                            'angle': self.HarmonicAngleForce[key][0],
                                                            'k': self.HarmonicAngleForce[key][1]})

        # add the torsion terms
        for key in self.PeriodicTorsionForce.keys():
            SubElement(PeriodicTorsionForce, "Proper", attrib={'class1': self.AtomTypes[key[0]][2],
                                                               'class2': self.AtomTypes[key[1]][2],
                                                               'class3': self.AtomTypes[key[2]][2],
                                                               'class4': self.AtomTypes[key[3]][2],
                                                               'k1': self.PeriodicTorsionForce[key][0][1],
                                                               'k2': self.PeriodicTorsionForce[key][1][1],
                                                               'k3': self.PeriodicTorsionForce[key][2][1],
                                                               'k4': self.PeriodicTorsionForce[key][3][1],
                                                               'periodicity1': '1', 'periodicity2': '2',
                                                               'periodicity3': '3', 'periodicity4': '4',
                                                               'phase1': self.PeriodicTorsionForce[key][0][2],
                                                               'phase2': self.PeriodicTorsionForce[key][1][2],
                                                               'phase3': self.PeriodicTorsionForce[key][2][2],
                                                               'phase4': self.PeriodicTorsionForce[key][3][2]})

        # add the non-bonded parameters
        for key in self.NonbondedForce.keys():
            SubElement(NonbondedForce, "Atom", attrib={'type': self.AtomTypes[key][1],
                                                       'charge': self.NonbondedForce[key][0],
                                                       'sigma': self.NonbondedForce[key][1],
                                                       'epsilon': self.NonbondedForce[key][2]})

        # Store the tree back into the molecule
        self.xml_tree = ElementTree(root)

    def read_xyz(self, name=None):
        """Read an xyz file to store the molecule structure."""

        opt_molecule = []

        # opt.xyz is the geometric optimised structure file.
        try:
            with open(f'{name if name else "opt"}.xyz', 'r') as xyz_file:
                lines = xyz_file.readlines()[2:]
                for line in lines:
                    line = line.split()
                    opt_molecule.append([line[0], float(line[1]), float(line[2]), float(line[3])])
            self.QMoptimized = opt_molecule

            return self

        except FileNotFoundError:
            raise FileNotFoundError('Cannot find xyz file to read.\nThis is likely due to PSI4 not generating one.\n'
                                    'Please ensure PSI4 is installed properly and can be called with the command: psi4\n'
                                    'Alternatively, geometric may not be installed properly.\n'
                                    'Please ensure it is and can be called with the command: geometric-optimize\n'
                                    'Installation instructions can be found on the respective github pages and '
                                    'elsewhere online, see README for more details.')

    def write_xyz(self, QM=False, MM=False, name=None):
        """Write a general xyz file. QM and MM decide where it will be written from in the ligand class."""

        if QM:
            molecule = self.QMoptimized
        elif MM:
            molecule = self.MMoptimized
        else:
            molecule = self.molecule

        with open(f'{name if name else self.name}.xyz', 'w+') as xyz_file:

            xyz_file.write(f'{len(molecule)}\n')
            xyz_file.write('xyz file generated with QUBEKit\n')

            for atom in molecule:
                # Format with spacing
                xyz_file.write(f'{atom[0]}       {atom[1]: .10f}   {atom[2]: .10f}   {atom[3]: .10f} \n')

    def pickle(self, state=None):
        """Pickles the ligand object in its current state to the (hidden) pickle file.
        If other pickle objects already exist for the particular ligand object:
        the latest object is put to the top.
        """

        mols = OrderedDict()
        # First check if the pickle file exists
        try:
            # Try to load a hidden pickle file; make sure to get all objects
            with open(f'.{self.name}_states', 'rb') as pickle_jar:
                while True:
                    try:
                        mol = load(pickle_jar)
                        mols[mol.state] = mol
                    except:
                        break
        except FileNotFoundError:
            pass

        # Now we can save the items; first assign the location
        self.state = state
        mols[self.state] = self

        # Open the pickle jar which will always be the ligand object's name
        with open(f'.{self.name}_states', 'wb') as pickle_jar:
            # If there were other molecules of the same state in the jar: overwrite them
            for key in mols.keys():
                dump(mols[key], pickle_jar)
