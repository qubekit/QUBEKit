#!/usr/bin/env python

from numpy import array, linalg, dot, degrees, cross, arctan2, arccos
from networkx import neighbors, Graph, has_path, draw
from matplotlib import pyplot as plt

from xml.etree.ElementTree import tostring, Element, SubElement, ElementTree
from xml.dom.minidom import parseString

from datetime import datetime
from pickle import dump, load
from re import sub
from collections import OrderedDict
from itertools import groupby, chain


class Molecule:
    """Base class for ligands and proteins."""

    def __init__(self, filename, smiles_string=None):
        """
        filename                str; Full filename e.g. methane.pdb
        name                    str; Molecule name e.g. methane
        smiles                  str; equal to the smiles_string if one is provided

        topology                Graph class object. Contains connection information for molecule
        molecule                List of lists; Inner list is the atom type followed by its coords
                                e.g. [['C', -0.022, 0.003, 0.017], ['H', -0.669, 0.889, -0.101], ...]
        angles                  Shows angles based on atom indices (+1) e.g. (1, 2, 4), (1, 2, 5)
        dihedrals
        rotatable
        atom_names
        bond_lengths
        dih_phis
        angle_values

        xml_tree
        AtomTypes               dict of lists; basic non-symmetrised atoms types for each atom in the molecule
                                e.g. {0, ['C1', 'opls_800', 'C800'], 1: ['H1', 'opls_801', 'H801'], ... }
        Residues
        HarmonicBondForce
        HarmonicAngleForce
        PeriodicTorsionForce
        NonbondedForce          OrderedDict; L-J params. Keys are atom index, vals are [charge, sigma, epsilon]

        log_file                str; Full log file name used by the run file in special run cases
        state
        """

        # Namings
        self.filename = filename
        self.name = filename[:-4]
        self.smiles = smiles_string

        # Structure
        self.topology = None
        self.molecule = None
        self.angles = None
        self.dihedrals = None
        self.rotatable = None
        self.atom_names = None
        self.bond_lengths = None
        self.dih_phis = None
        self.angle_values = None

        # XML Info
        self.xml_tree = None
        self.AtomTypes = {}
        self.Residues = {}
        self.HarmonicBondForce = {}
        self.HarmonicAngleForce = {}
        self.PeriodicTorsionForce = OrderedDict()
        self.NonbondedForce = OrderedDict()

        # QUBEKit internals
        self.log_file = None
        self.state = None

        # Atomic weight dict
        self.element_dict = {'H': 1.008000,  # Group 1
                             'C': 12.011000,  # Group 4
                             'N': 14.007000, 'P': 30.973762,  # Group 5
                             'O': 15.999000, 'S': 32.060000,  # Group 6
                             'F': 18.998403, 'Cl': 35.450000, 'Br': 79.904000, 'I': 126.904470  # Group 7
                             }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def read_pdb(self, qm=False, mm=False):
        """
        Reads the input PDB file to find the ATOM or HETATM tags, extracts the elements and xyz coordinates.
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

        if qm:
            self.qm_optimised = molecule
        elif mm:
            self.mm_optimised = molecule
        else:
            self.molecule = molecule

        return self

    def find_angles(self):
        """
        Take the topology graph network and return a list of all angle combinations.
        Checked against OPLS-AA on molecules containing 10-63 angles.
        """

        self.angles = []

        for node in self.topology.nodes:
            bonded = sorted(list(neighbors(self.topology, node)))

            # Check that the atom has more than one bond
            if len(bonded) < 2:
                continue

            # Find all possible angle combinations from the list
            for i in range(len(bonded)):
                for j in range(i + 1, len(bonded)):
                    atom1, atom3 = bonded[i], bonded[j]

                    self.angles.append((atom1, node, atom3))

        return self.angles

    def get_bond_lengths(self, qm=False, mm=False):
        """For the given molecule and topology find the length of all of the bonds."""

        self.bond_lengths = {}

        if qm:
            molecule = self.qm_optimised
        elif mm:
            molecule = self.mm_optimised
        else:
            molecule = self.molecule

        for edge in self.topology.edges:
            atom1 = array(molecule[int(edge[0]) - 1][1:])
            atom2 = array(molecule[int(edge[1]) - 1][1:])
            bond_dist = linalg.norm(atom2 - atom1)
            self.bond_lengths[edge] = bond_dist

    def find_dihedrals(self):
        """
        Take the topology graph network and again return a dictionary of all possible dihedral combinations stored under
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

                            if edge not in self.dihedrals:
                                # Add the central edge as a key the first time it is used
                                self.dihedrals[edge] = [(start, edge[0], edge[1], end)]

                            else:
                                # Add the tuple to the correct key.
                                self.dihedrals[edge].append((start, edge[0], edge[1], end))

    def find_rotatable_dihedrals(self):
        """
        For each dihedral in the topology graph network and dihedrals dictionary, work out if the torsion is
        rotatable. Returns a list of dihedral dictionary keys representing the rotatable dihedrals.
        """

        self.rotatable = []

        # For each dihedral key remove the edge from the network
        for key in self.dihedrals:
            self.topology.remove_edge(*key)

            # Check if there is still a path between the two atoms in the edges.
            if not has_path(self.topology, key[0], key[1]):
                self.rotatable.append(key)

            # Add edge back to the network and try next key
            self.topology.add_edge(*key)

    def get_dihedral_values(self, qm=False, mm=False):
        """
        Taking the molecules' xyz coordinates and dihedrals dictionary, return a dictionary of dihedral
        angle keys and values. Also an option to only supply the keys of the dihedrals you want to calculate.
        """

        self.dih_phis = {}

        # Check if a rotatable tuple list is supplied, else calculate the angles for all dihedrals in the molecule.
        keys = self.rotatable if self.rotatable else list(self.dihedrals.keys())

        if qm:
            molecule = self.qm_optimised
        elif mm:
            molecule = self.mm_optimised
        else:
            molecule = self.molecule

        for key in keys:
            for torsion in self.dihedrals[key]:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1, x2, x3, x4 = [array(molecule[int(torsion[i]) - 1][1:]) for i in range(4)]
                b1, b2, b3 = x2 - x1, x3 - x2, x4 - x3
                t1 = linalg.norm(b2) * dot(b1, cross(b2, b3))
                t2 = dot(cross(b1, b2), cross(b2, b3))
                self.dih_phis[torsion] = degrees(arctan2(t1, t2))

    def get_angle_values(self, qm=False, mm=False):
        """
        For the given molecule and list of angle terms measure the angle values,
        then return a dictionary of angles and values.
        """

        self.angle_values = {}

        if qm:
            molecule = self.qm_optimised
        elif mm:
            molecule = self.mm_optimised
        else:
            molecule = self.molecule

        for angle in self.angles:
            x1 = array(molecule[int(angle[0]) - 1][1:])
            x2 = array(molecule[int(angle[1]) - 1][1:])
            x3 = array(molecule[int(angle[2]) - 1][1:])
            b1, b2 = x1 - x2, x3 - x2
            cosine_angle = dot(b1, b2) / (linalg.norm(b1) * linalg.norm(b2))
            self.angle_values[angle] = degrees(arccos(cosine_angle))

    def write_pdb(self, qm=False, mm=False, name=None):
        """
        Take the current molecule and topology and write a pdb file for the molecule.
        Only for small molecules, not standard residues. No size limit.
        """

        if qm:
            molecule = self.qm_optimised
        elif mm:
            molecule = self.mm_optimised
        else:
            molecule = self.molecule

        with open(f'{name if name is not None else self.name}.pdb', 'w+') as pdb_file:

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

        with open(f'{name if name is not None else self.name}.xml', 'w+') as xml_doc:
            xml_doc.write(pretty_xml_as_string)

    def build_tree(self):
        """Separates the parameters and builds an xml tree ready to be used."""

        # Create XML layout
        root = Element('ForceField')
        AtomTypes = SubElement(root, "AtomTypes")
        Residues = SubElement(root, "Residues")
        Residue = SubElement(Residues, "Residue", name="UNK")
        HarmonicBondForce = SubElement(root, "HarmonicBondForce")
        HarmonicAngleForce = SubElement(root, "HarmonicAngleForce")
        PeriodicTorsionForce = SubElement(root, "PeriodicTorsionForce")
        NonbondedForce = SubElement(root, "NonbondedForce", attrib={'coulomb14scale': "0.5", 'lj14scale': "0.5"})

        for key, val in self.AtomTypes.items():
            SubElement(AtomTypes, "Type", attrib={'name': val[1], 'class': val[2],
                                                  'element': self.molecule[key][0],
                                                  'mass': str(self.element_dict[self.molecule[key][0]])})

            SubElement(Residue, "Atom", attrib={'name': val[0], 'type': val[1]})

        # Add the bonds / connections
        for key, val in self.HarmonicBondForce.items():
            SubElement(Residue, "Bond", attrib={'from': str(key[0]), 'to': str(key[1])})

            SubElement(HarmonicBondForce, "Bond", attrib={'class1': self.AtomTypes[key[0]][2],
                                                          'class2': self.AtomTypes[key[1]][2],
                                                          'length': val[0], 'k': val[1]})

        # Add the angles
        for key, val in self.HarmonicAngleForce.items():
            SubElement(HarmonicAngleForce, "Angle", attrib={'class1': self.AtomTypes[key[0]][2],
                                                            'class2': self.AtomTypes[key[1]][2],
                                                            'class3': self.AtomTypes[key[2]][2],
                                                            'angle': val[0], 'k': val[1]})

        # Add the torsion terms
        for key, val in self.PeriodicTorsionForce.items():
            SubElement(PeriodicTorsionForce, "Proper", attrib={'class1': self.AtomTypes[key[0]][2],
                                                               'class2': self.AtomTypes[key[1]][2],
                                                               'class3': self.AtomTypes[key[2]][2],
                                                               'class4': self.AtomTypes[key[3]][2],
                                                               'k1': val[0][1], 'k2': val[1][1],
                                                               'k3': val[2][1], 'k4': val[3][1],
                                                               'periodicity1': '1', 'periodicity2': '2',
                                                               'periodicity3': '3', 'periodicity4': '4',
                                                               'phase1': val[0][2], 'phase2': val[1][2],
                                                               'phase3': val[2][2], 'phase4': val[3][2]})

        # Add the non-bonded parameters
        for key, val in self.NonbondedForce.items():
            SubElement(NonbondedForce, "Atom", attrib={'type': self.AtomTypes[key][1], 'charge': val[0],
                                                       'sigma': val[1], 'epsilon': val[2]})

        # Store the tree back into the molecule
        self.xml_tree = ElementTree(root)

    def write_xyz(self, qm=False, mm=False, name=None):
        """Write a general xyz file. QM and MM decide where it will be written from in the ligand class."""

        if qm:
            molecule = self.qm_optimised
        elif mm:
            molecule = self.mm_optimised
        else:
            molecule = self.molecule

        with open(f'{name if name is not None else self.name}.xyz', 'w+') as xyz_file:

            xyz_file.write(f'{len(molecule)}\n')
            xyz_file.write('xyz file generated with QUBEKit\n')

            for atom in molecule:
                # Format with spacing
                xyz_file.write(f'{atom[0]}       {atom[1]: .10f}   {atom[2]: .10f}   {atom[3]: .10f} \n')

    def write_gromacs_file(self):
        """To a gromacs file, write and format the necessary variables."""

        with open(f'{self.name}.gro', 'w+') as gro_file:
            gro_file.write(f'NEW {self.name.upper()} GRO FILE\n')
            gro_file.write(f'{len(self.molecule):>5}\n')
            for pos, atom in enumerate(self.molecule, 1):
                # 'mol number''mol name'  'atom name'   'atom count'   'x coord'   'y coord'   'z coord'
                # 1WATER  OW1    1   0.126   1.624   1.679
                gro_file.write(f'    1{self.name.upper()}  {atom[0]}{pos}   {pos}   {atom[1]: .3f}   {atom[2]: .3f}   {atom[3]: .3f}\n')

    def pickle(self, state=None):
        """
        Pickles the Molecule object in its current state to the (hidden) pickle file.
        If other pickle objects already exist for the particular object:
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
            for val in mols.values():
                dump(val, pickle_jar)


class Ligand(Molecule):

    def __init__(self, filename, smiles_string=None):
        """
        scan_order
        mm_optimised
        qm_optimised
        parameter_engine
        hessian                 2d numpy array; matrix of size 3N x 3N where N is number of atoms in the molecule
        modes
        QM_scan_energy
        MM_scan_energy
        descriptors
        symmetry_types          list; symmetrised atom types
        """

        super().__init__(filename, smiles_string)

        self.scan_order = None
        self.mm_optimised = None
        self.qm_optimised = None
        self.parameter_engine = None
        self.hessian = None
        self.modes = None

        self.QM_scan_energy = {}
        self.MM_scan_energy = {}
        self.descriptors = {}
        self.symmetry_types = []

        self.read_pdb()
        self.find_angles()
        self.find_dihedrals()
        self.find_rotatable_dihedrals()
        self.get_dihedral_values()
        self.get_bond_lengths()
        self.get_angle_values()

    def __str__(self, trunc=False):
        """
        Prints the ligand class objects' names and values one after another with new lines between each.
        Mostly just used for logging and displaying the results at the end of a run.
        If trunc is set to True:
            Check the items being printed:
                If None -> print them as normal
                If they're short (<120 chars) -> print them as normal
                Otherwise -> print a truncated version of them.
        This is called with:   Ligand(filename='').__str__(trunc=True)
        """

        # This is the old __str__ definition which is basically a one-line alternative to the else case below.
        # return '\n'.join(('{} = {}'.format(key, val) for key, val in self.__dict__.items()))

        return_str = ''
        for key, val in self.__dict__.items():
            if trunc:
                # if it's smaller than 120 chars: print it as is. Otherwise print a truncated version.
                return_str += f'\n{key} = {val if (len(str(key) + str(val)) < 120) else str(val)[:121 - len(str(key))] + "..."}'
            else:
                # Return all objects as {ligand object name} = {ligand object value(s)} without any special formatting.
                return_str += f'\n{key} = {val}\n'

        return return_str

    def read_xyz(self, name=None):
        """Read an xyz file to store the molecule structure."""

        opt_molecule = []

        # opt.xyz is the geometric optimised structure file.
        try:
            with open(f'{name if name is not None else "opt"}.xyz', 'r') as xyz_file:
                lines = xyz_file.readlines()[2:]
                for line in lines:
                    line = line.split()
                    opt_molecule.append([line[0], float(line[1]), float(line[2]), float(line[3])])
            self.qm_optimised = opt_molecule

        except FileNotFoundError:
            raise FileNotFoundError('Cannot find xyz file to read.\nThis is likely due to PSI4 not generating one.\n'
                                    'Please ensure PSI4 is installed properly and can be called with the command: psi4\n'
                                    'Alternatively, geometric may not be installed properly.\n'
                                    'Please ensure it is and can be called with the command: geometric-optimize\n'
                                    'Installation instructions can be found on the respective github pages and '
                                    'elsewhere online, see README for more details.')


class Protein(Molecule):
    """
    Class to handle proteins as chains of amino acids.
    Dicts are used to provide any standard parameters;
    the orders of the residues and atoms in the residues is constructed from the pdb file provided.
    """

    def __init__(self, filename):
        """
        order                   list of str; list of the residues in the protein, in order.
        bonds                   list of tuples of two ints; describes each bond in the molecule where the ints are the atom indices.

        bonds_dict              dict, key=str, val=list of tuples; key=residue name: val=list of the bond connections which are tuples
                                e.g. {'leu': [(0, 1), (1, 2), ...], ...}
        externals               dict, key=str, val=tuple, int or None; Similar to bonds_dict except this is only for the external bonds.
                                Some keys will have two external bonds, caps only have one, and others (e.g. ions) don't have any.

        n_atoms                 dict; key=str, val=int; key=residue name: val=number of atoms in that residue.

        bond_constants
        angles_dict
        propers_dict
        impropers_dict
        """

        super().__init__(filename)

        self.order = None
        self.bonds = None

        self.bonds_dict = {'ace': [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5)],
                           'ala': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 8), (4, 5), (4, 6), (4, 7), (8, 9)],
                           'arg': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 22), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (10, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18), (19, 20), (19, 21),
                                   (22, 23)],
                           'ash': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 11), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (9, 10), (11, 12)],
                           'asn': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 12), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (9, 10), (9, 11),
                                   (12, 13)],
                           'asp': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 10), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (10, 11)],
                           'cala': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 8), (4, 5), (4, 6), (4, 7), (8, 9), (8, 10)],
                           'carg': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 22), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (10, 12), (10, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18), (19, 20), (19, 21),
                                    (22, 23), (22, 24)],
                           'casn': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 12), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (9, 10), (9, 11),
                                    (12, 13), (12, 14)],
                           'casp': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 10), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (10, 11), (10, 12)],
                           'ccys': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 9), (4, 5), (4, 6), (4, 7), (7, 8), (9, 10), (9, 11)],
                           'ccyx': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 8), (4, 5), (4, 6), (4, 7), (8, 9), (8, 10)],
                           'cgln': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (10, 12), (12, 13), (12, 14), (15, 16), (15, 17)],
                           'cglu': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 13), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (10, 12), (13, 14), (13, 15)],
                           'cgly': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7)],
                           'chid': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 13), (8, 9), (8, 10),
                                    (10, 11), (10, 12), (12, 13), (13, 14), (15, 16), (15, 17)],
                           'chie': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 13), (8, 9), (9, 10),
                                    (9, 11), (11, 12), (11, 13), (13, 14), (15, 16), (15, 17)],
                           'chip': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 16), (4, 5), (4, 6), (4, 7), (7, 8), (7, 14), (8, 9), (8, 10),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (16, 17), (16, 18)],
                           'cile': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 17), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                    (10, 12), (10, 13), (13, 14), (13, 15), (13, 16), (17, 18), (17, 19)],
                           'cleu': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 17), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 13), (9, 10),
                                    (9, 11), (9, 12), (13, 14), (13, 15), (13, 16), (17, 18), (17, 19)],
                           'clys': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 20), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (10, 12), (10, 13), (13, 14), (13, 15), (13, 16), (16, 17), (16, 18), (16, 19), (20, 21), (20, 22)],
                           'cmet': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (11, 12), (11, 13), (11, 14), (15, 16), (15, 17)],
                           'cphe': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 18), (4, 5), (4, 6), (4, 7), (7, 8), (7, 16), (8, 9), (8, 10),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (18, 19), (18, 20)],
                           'cpro': [(0, 1), (0, 10), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                    (10, 12), (12, 13), (12, 14)],
                           'cser': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 9), (4, 5), (4, 6), (4, 7), (7, 8), (9, 10), (9, 11)],
                           'cthr': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 12), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                    (12, 13), (12, 14)],
                           'ctrp': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 22), (4, 5), (4, 6), (4, 7), (7, 8), (7, 21), (8, 9), (8, 10),
                                    (10, 11), (10, 12), (12, 13), (12, 21), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                    (19, 20), (19, 21), (22, 23), (22, 24)],
                           'ctyr': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 19), (4, 5), (4, 6), (4, 7), (7, 8), (7, 17), (8, 9), (8, 10),
                                    (10, 11), (10, 12), (12, 13), (12, 15), (13, 14), (15, 16), (15, 17), (17, 18), (19, 20), (19, 21)],
                           'cval': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 14), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                    (10, 12), (10, 13), (14, 15), (14, 16)],
                           'cym': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 8), (4, 5), (4, 6), (4, 7), (8, 9)],
                           'cys': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 9), (4, 5), (4, 6), (4, 7), (7, 8), (9, 10)],
                           'cyx': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 8), (4, 5), (4, 6), (4, 7), (8, 9)],
                           'cl-': [],
                           'cs+': [],
                           'da': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                  (10, 12), (10, 28), (12, 13), (12, 25), (13, 14), (13, 15), (15, 16), (16, 17), (16, 25), (17, 18),
                                  (17, 21), (18, 19), (18, 20), (21, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 31),
                                  (28, 29), (28, 30)],
                           'da3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                   (10, 12), (10, 28), (12, 13), (12, 25), (13, 14), (13, 15), (15, 16), (16, 17), (16, 25), (17, 18),
                                   (17, 21), (18, 19), (18, 20), (21, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 31),
                                   (28, 29), (28, 30), (31, 32)],
                           'da5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 23), (11, 12), (11, 13), (13, 14), (14, 15), (14, 23), (15, 16), (15, 19), (16, 17),
                                   (16, 18), (19, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28)],
                           'dan': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 23), (11, 12), (11, 13), (13, 14), (14, 15), (14, 23), (15, 16), (15, 19), (16, 17),
                                   (16, 18), (19, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28),
                                   (29, 30)],
                           'dc': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 24), (9, 10), (10, 11),
                                  (10, 12), (10, 26), (12, 13), (12, 22), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 21),
                                  (18, 19), (18, 20), (21, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28)],
                           'dc3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 24), (9, 10), (10, 11),
                                   (10, 12), (10, 26), (12, 13), (12, 22), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 21),
                                   (18, 19), (18, 20), (21, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28), (29, 30)],
                           'dc5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 22), (7, 8), (8, 9), (8, 10), (8, 24),
                                   (10, 11), (10, 20), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18),
                                   (19, 20), (20, 21), (22, 23), (22, 24), (22, 27), (24, 25), (24, 26)],
                           'dcn': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 22), (7, 8), (8, 9), (8, 10), (8, 24),
                                   (10, 11), (10, 20), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18),
                                   (19, 20), (20, 21), (22, 23), (22, 24), (22, 27), (24, 25), (24, 26), (27, 28)],
                           'dg': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 27), (9, 10), (10, 11),
                                  (10, 12), (10, 29), (12, 13), (12, 26), (13, 14), (13, 15), (15, 16), (16, 17), (16, 26), (17, 18),
                                  (17, 19), (19, 20), (19, 21), (21, 22), (21, 25), (22, 23), (22, 24), (25, 26), (27, 28), (27, 29),
                                  (27, 32), (29, 30), (29, 31)],
                           'dg3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 27), (9, 10), (10, 11),
                                   (10, 12), (10, 29), (12, 13), (12, 26), (13, 14), (13, 15), (15, 16), (16, 17), (16, 26), (17, 18),
                                   (17, 19), (19, 20), (19, 21), (21, 22), (21, 25), (22, 23), (22, 24), (25, 26), (27, 28), (27, 29),
                                   (27, 32), (29, 30), (29, 31), (32, 33)],
                           'dg5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 25), (7, 8), (8, 9), (8, 10), (8, 27),
                                   (10, 11), (10, 24), (11, 12), (11, 13), (13, 14), (14, 15), (14, 24), (15, 16), (15, 17), (17, 18),
                                   (17, 19), (19, 20), (19, 23), (20, 21), (20, 22), (23, 24), (25, 26), (25, 27), (25, 30), (27, 28),
                                   (27, 29)],
                           'dgn': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 25), (7, 8), (8, 9), (8, 10), (8, 27),
                                   (10, 11), (10, 24), (11, 12), (11, 13), (13, 14), (14, 15), (14, 24), (15, 16), (15, 17), (17, 18),
                                   (17, 19), (19, 20), (19, 23), (20, 21), (20, 22), (23, 24), (25, 26), (25, 27), (25, 30), (27, 28),
                                   (27, 29), (30, 31)],
                           'dt': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                  (10, 12), (10, 28), (12, 13), (12, 24), (13, 14), (13, 15), (15, 16), (15, 20), (16, 17), (16, 18),
                                  (16, 19), (20, 21), (20, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 31), (28, 29),
                                  (28, 30)],
                           'dt3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                   (10, 12), (10, 28), (12, 13), (12, 24), (13, 14), (13, 15), (15, 16), (15, 20), (16, 17), (16, 18),
                                   (16, 19), (20, 21), (20, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 31), (28, 29),
                                   (28, 30), (31, 32)],
                           'dt5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 22), (11, 12), (11, 13), (13, 14), (13, 18), (14, 15), (14, 16), (14, 17), (18, 19),
                                   (18, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28)],
                           'dtn': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 22), (11, 12), (11, 13), (13, 14), (13, 18), (14, 15), (14, 16), (14, 17), (18, 19),
                                   (18, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 29), (26, 27), (26, 28), (29, 30)],
                           'glh': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 14), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (12, 13), (14, 15)],
                           'gln': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (12, 13), (12, 14), (15, 16)],
                           'glu': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 13), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (13, 14)],
                           'gly': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 5), (5, 6)],
                           'hid': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 13), (8, 9), (8, 10),
                                   (10, 11), (10, 12), (12, 13), (13, 14), (15, 16)],
                           'hie': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 13), (8, 9), (9, 10),
                                   (9, 11), (11, 12), (11, 13), (13, 14), (15, 16)],
                           'hip': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 16), (4, 5), (4, 6), (4, 7), (7, 8), (7, 14), (8, 9), (8, 10),
                                   (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (16, 17)],
                           'ile': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 17), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                   (10, 12), (10, 13), (13, 14), (13, 15), (13, 16), (17, 18)],
                           'k+': [],
                           'leu': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 17), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 13), (9, 10),
                                   (9, 11), (9, 12), (13, 14), (13, 15), (13, 16), (17, 18)],
                           'lyn': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 19), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (10, 13), (13, 14), (13, 15), (13, 16), (16, 17), (16, 18), (19, 20)],
                           'lys': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 20), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (10, 13), (13, 14), (13, 15), (13, 16), (16, 17), (16, 18), (16, 19), (20, 21)],
                           'li+': [],
                           'met': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 15), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (11, 12), (11, 13), (11, 14), (15, 16)],
                           'mg2': [],
                           'nala': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11)],
                           'narg': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 24), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (12, 14), (12, 15), (15, 16), (15, 17), (17, 18), (17, 21), (18, 19), (18, 20),
                                    (21, 22), (21, 23), (24, 25)],
                           'nasn': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 14), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (11, 12), (11, 13), (14, 15)],
                           'nasp': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 12), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (12, 13)],
                           'ncys': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 11), (6, 7), (6, 8), (6, 9), (9, 10), (11, 12)],
                           'ncyx': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11)],
                           'ngln': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 17), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (12, 14), (14, 15), (14, 16), (17, 18)],
                           'nglu': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 15), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (12, 14), (15, 16)],
                           'ngly': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8)],
                           'nhe': [(0, 1), (0, 2)],
                           'nhid': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 17), (6, 7), (6, 8), (6, 9), (9, 10), (9, 15),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (15, 16), (17, 18)],
                           'nhie': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 17), (6, 7), (6, 8), (6, 9), (9, 10), (9, 15),
                                    (10, 11), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (17, 18)],
                           'nhip': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 18), (6, 7), (6, 8), (6, 9), (9, 10), (9, 16),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (18, 19)],
                           'nile': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 19), (6, 7), (6, 8), (6, 12), (8, 9), (8, 10),
                                    (8, 11), (12, 13), (12, 14), (12, 15), (15, 16), (15, 17), (15, 18), (19, 20)],
                           'nleu': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 19), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 15), (11, 12), (11, 13), (11, 14), (15, 16), (15, 17), (15, 18), (19, 20)],
                           'nlys': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 22), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (12, 14), (12, 15), (15, 16), (15, 17), (15, 18), (18, 19), (18, 20), (18, 21),
                                    (22, 23)],
                           'nme': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 5)],
                           'nmet': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 17), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (13, 14), (13, 15), (13, 16), (17, 18)],
                           'nphe': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 20), (6, 7), (6, 8), (6, 9), (9, 10), (9, 18),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (16, 18), (18, 19), (20, 21)],
                           'npro': [(0, 1), (0, 2), (0, 3), (0, 12), (3, 4), (3, 5), (3, 6), (6, 7), (6, 8), (6, 9), (9, 10), (9, 11),
                                    (9, 12), (12, 13), (12, 14), (14, 15)],
                           'nser': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 11), (6, 7), (6, 8), (6, 9), (9, 10), (11, 12)],
                           'nthr': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 14), (6, 7), (6, 8), (6, 12), (8, 9), (8, 10),
                                    (8, 11), (12, 13), (14, 15)],
                           'ntrp': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 24), (6, 7), (6, 8), (6, 9), (9, 10), (9, 23),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 23), (15, 16), (15, 17), (17, 18), (17, 19),
                                    (19, 20), (19, 21), (21, 22), (21, 23), (24, 25)],
                           'ntyr': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 21), (6, 7), (6, 8), (6, 9), (9, 10), (9, 19),
                                    (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 17), (15, 16), (17, 18), (17, 19), (19, 20),
                                    (21, 22)],
                           'nval': [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 16), (6, 7), (6, 8), (6, 12), (8, 9), (8, 10),
                                    (8, 11), (12, 13), (12, 14), (12, 15), (16, 17)],
                           'na+': [],
                           'phe': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 18), (4, 5), (4, 6), (4, 7), (7, 8), (7, 16), (8, 9), (8, 10),
                                   (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (18, 19)],
                           'pro': [(0, 1), (0, 10), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11),
                                   (10, 12), (12, 13)],
                           'ra': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                  (10, 12), (10, 28), (12, 13), (12, 25), (13, 14), (13, 15), (15, 16), (16, 17), (16, 25), (17, 18),
                                  (17, 21), (18, 19), (18, 20), (21, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 32),
                                  (28, 29), (28, 30), (30, 31)],
                           'ra3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 26), (9, 10), (10, 11),
                                   (10, 12), (10, 28), (12, 13), (12, 25), (13, 14), (13, 15), (15, 16), (16, 17), (16, 25), (17, 18),
                                   (17, 21), (18, 19), (18, 20), (21, 22), (22, 23), (22, 24), (24, 25), (26, 27), (26, 28), (26, 32),
                                   (28, 29), (28, 30), (30, 31), (32, 33)],
                           'ra5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 23), (11, 12), (11, 13), (13, 14), (14, 15), (14, 23), (15, 16), (15, 19), (16, 17),
                                   (16, 18), (19, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 30), (26, 27), (26, 28),
                                   (28, 29)],
                           'ran': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 24), (7, 8), (8, 9), (8, 10), (8, 26),
                                   (10, 11), (10, 23), (11, 12), (11, 13), (13, 14), (14, 15), (14, 23), (15, 16), (15, 19), (16, 17),
                                   (16, 18), (19, 20), (20, 21), (20, 22), (22, 23), (24, 25), (24, 26), (24, 30), (26, 27), (26, 28),
                                   (28, 29), (30, 31)],
                           'rc': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 24), (9, 10), (10, 11),
                                  (10, 12), (10, 26), (12, 13), (12, 22), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 21),
                                  (18, 19), (18, 20), (21, 22), (22, 23), (24, 25), (24, 26), (24, 30), (26, 27), (26, 28), (28, 29)],
                           'rc3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 24), (9, 10), (10, 11),
                                   (10, 12), (10, 26), (12, 13), (12, 22), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 21),
                                   (18, 19), (18, 20), (21, 22), (22, 23), (24, 25), (24, 26), (24, 30), (26, 27), (26, 28), (28, 29),
                                   (30, 31)],
                           'rc5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 22), (7, 8), (8, 9), (8, 10), (8, 24),
                                   (10, 11), (10, 20), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18),
                                   (19, 20), (20, 21), (22, 23), (22, 24), (22, 28), (24, 25), (24, 26), (26, 27)],
                           'rcn': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 22), (7, 8), (8, 9), (8, 10), (8, 24),
                                   (10, 11), (10, 20), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 19), (16, 17), (16, 18),
                                   (19, 20), (20, 21), (22, 23), (22, 24), (22, 28), (24, 25), (24, 26), (26, 27), (28, 29)],
                           'rg': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 27), (9, 10), (10, 11),
                                  (10, 12), (10, 29), (12, 13), (12, 26), (13, 14), (13, 15), (15, 16), (16, 17), (16, 26), (17, 18),
                                  (17, 19), (19, 20), (19, 21), (21, 22), (21, 25), (22, 23), (22, 24), (25, 26), (27, 28), (27, 29),
                                  (27, 33), (29, 30), (29, 31), (31, 32)],
                           'rg3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 27), (9, 10), (10, 11),
                                   (10, 12), (10, 29), (12, 13), (12, 26), (13, 14), (13, 15), (15, 16), (16, 17), (16, 26), (17, 18),
                                   (17, 19), (19, 20), (19, 21), (21, 22), (21, 25), (22, 23), (22, 24), (25, 26), (27, 28), (27, 29),
                                   (27, 33), (29, 30), (29, 31), (31, 32), (33, 34)],
                           'rg5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 25), (7, 8), (8, 9), (8, 10), (8, 27),
                                   (10, 11), (10, 24), (11, 12), (11, 13), (13, 14), (14, 15), (14, 24), (15, 16), (15, 17), (17, 18),
                                   (17, 19), (19, 20), (19, 23), (20, 21), (20, 22), (23, 24), (25, 26), (25, 27), (25, 31), (27, 28),
                                   (27, 29), (29, 30)],
                           'rgn': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 25), (7, 8), (8, 9), (8, 10), (8, 27),
                                   (10, 11), (10, 24), (11, 12), (11, 13), (13, 14), (14, 15), (14, 24), (15, 16), (15, 17), (17, 18),
                                   (17, 19), (19, 20), (19, 23), (20, 21), (20, 22), (23, 24), (25, 26), (25, 27), (25, 31), (27, 28),
                                   (27, 29), (29, 30), (31, 32)],
                           'ru': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 23), (9, 10), (10, 11),
                                  (10, 12), (10, 25), (12, 13), (12, 21), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                  (19, 20), (19, 21), (21, 22), (23, 24), (23, 25), (23, 29), (25, 26), (25, 27), (27, 28)],
                           'ru3': [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 23), (9, 10), (10, 11),
                                   (10, 12), (10, 25), (12, 13), (12, 21), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                   (19, 20), (19, 21), (21, 22), (23, 24), (23, 25), (23, 29), (25, 26), (25, 27), (27, 28), (29, 30)],
                           'ru5': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 21), (7, 8), (8, 9), (8, 10), (8, 23),
                                   (10, 11), (10, 19), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                   (19, 20), (21, 22), (21, 23), (21, 27), (23, 24), (23, 25), (25, 26)],
                           'run': [(0, 1), (1, 2), (2, 3), (2, 4), (2, 5), (5, 6), (5, 7), (5, 21), (7, 8), (8, 9), (8, 10), (8, 23),
                                   (10, 11), (10, 19), (11, 12), (11, 13), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                   (19, 20), (21, 22), (21, 23), (21, 27), (23, 24), (23, 25), (25, 26), (27, 28)],
                           'rb+': [],
                           'ser': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 9), (4, 5), (4, 6), (4, 7), (7, 8), (9, 10)],
                           'thr': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 12), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                   (12, 13)],
                           'trp': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 22), (4, 5), (4, 6), (4, 7), (7, 8), (7, 21), (8, 9), (8, 10),
                                   (10, 11), (10, 12), (12, 13), (12, 21), (13, 14), (13, 15), (15, 16), (15, 17), (17, 18), (17, 19),
                                   (19, 20), (19, 21), (22, 23)],
                           'tyr': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 19), (4, 5), (4, 6), (4, 7), (7, 8), (7, 17), (8, 9), (8, 10),
                                   (10, 11), (10, 12), (12, 13), (12, 15), (13, 14), (15, 16), (15, 17), (17, 18), (19, 20)],
                           'val': [(0, 1), (0, 2), (2, 3), (2, 4), (2, 14), (4, 5), (4, 6), (4, 10), (6, 7), (6, 8), (6, 9), (10, 11),
                                   (10, 12), (10, 13), (14, 15)]}

        # Ordering of the tuples is (N, C). For the caps, whether or not it's a Nitrogen or Carbon atom is implicit.
        self.externals = {'ace': 4, 'ala': (0, 8), 'arg': (0, 22), 'ash': (0, 11), 'asn': (0, 12), 'asp': (0, 10), 'cala': 0, 'carg': 0,
                          'casn': 0, 'casp': 0, 'ccys': 0, 'ccyx': (0, 7), 'cgln': 0, 'cglu': 0, 'cgly': 0, 'chid': 0, 'chie': 0, 'chip': 0,
                          'cile': 0, 'cleu': 0, 'clys': 0, 'cmet': 0, 'cphe': 0, 'cpro': 0, 'cser': 0, 'cthr': 0, 'ctrp': 0, 'ctyr': 0,
                          'cval': 0, 'cym': (0, 8), 'cys': (0, 9), 'cyx': (0, 8), 'cl-': None, 'cs+': None, 'da': (0, 31), 'da3': 0,
                          'da5': 29, 'dan': None, 'dc': (0, 29), 'dc3': 0, 'dc5': 27, 'dcn': None, 'dg': (0, 32), 'dg3': 0, 'dg5': 30,
                          'dgn': None, 'dt': (0, 31), 'dt3': 0, 'dt5': 29, 'dtn': None, 'glh': (0, 14), 'gln': (0, 15), 'glu': (0, 13),
                          'gly': (0, 5), 'hid': (0, 15), 'hie': (0, 15), 'hip': (0, 16), 'ile': (0, 17), 'k+': None, 'leu': (0, 17),
                          'lyn': (0, 19), 'lys': (0, 20), 'li+': None, 'met': (0, 15), 'mg2': None, 'nala': 10, 'narg': 24, 'nasn': 14,
                          'nasp': 12, 'ncys': 11, 'ncyx': (10, 9), 'ngln': 17, 'nglu': 15, 'ngly': 7, 'nhe': 0, 'nhid': 17, 'nhie': 17,
                          'nhip': 18, 'nile': 19, 'nleu': 19, 'nlys': 22, 'nme': 0, 'nmet': 17, 'nphe': 20, 'npro': 14, 'nser': 11,
                          'nthr': 14, 'ntrp': 24, 'ntyr': 21, 'nval': 16, 'na+': None, 'phe': (0, 18), 'pro': (0, 12), 'ra': (0, 32),
                          'ra3': 0, 'ra5': 30, 'ran': None, 'rc': (0, 30), 'rc3': 0, 'rc5': 28, 'rcn': None, 'rg': (0, 33), 'rg3': 0,
                          'rg5': 31, 'rgn': None, 'ru': (0, 29), 'ru3': 0, 'ru5': 27, 'run': None, 'rb+': None, 'ser': (0, 9),
                          'thr': (0, 12), 'trp': (0, 22), 'tyr': (0, 19), 'val': (0, 14)}

        # Find the number of atoms in each residue; find the max value from the list of tuples (0 if the list of tuples is empty)
        # Because this finds the max index from the tuples, +1 is needed to get the actual number of atoms
        self.n_atoms = {k: max(chain.from_iterable(v)) + 1 if v else 0 for k, v in self.bonds_dict.items()}

        # self.bond_constants = {'res': {(0, 1): [0.1011, 235634567], ...}, ...}

        # self.angles_dict = {'res': {(0, 1, 2): [2.345, 456345], ...}, ...}

        # List is the force constants
        # self.propers_dict = {'res': {(0, 1, 2, 3): [345, 432345, 34573, 25646], ...}, ...}

        # self.impropers_dict = {'res': {(2, 0, 1, 3): [234, 456, 3467, 6245], ...}, ...}

        self.get_aa_order()

    def get_aa_order(self):
        """From a protein pdb file, extract the residue order (self.order)."""

        with open(self.filename, 'r') as pdb:
            lines = pdb.readlines()

        all_residues = []

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                all_residues.append(line.split()[3].lower())

        self.order = [res for res, group in groupby(all_residues)]

    def identify_bonds(self):
        """
        Stitch together the residues' parameters from the xml file.

        For each residue add the external bond;
        this will be tuple[1] for the first residue and the tuple[0] for the next residue.
        Then add the internal bonds.
        All bonds will be incremented by however many atoms came before them in the chain.
        """

        self.bonds = []

        # Append the cap's bonds to the bonds list
        self.bonds.extend(self.bonds_dict[self.order[0]])
        # Identify the current capping atom (the atom where the cap attaches to the protein)
        current_cap = self.externals[self.order[0]]
        # atom_count isn't always just the current_cap; sometimes the highest index atom is on a sidechain
        atom_count = self.n_atoms[self.order[0]]

        # Start main loop, checking through all the chained residues (not the caps)
        for res in self.order[1:-1]:
            # Construct the tuple for the external bond
            incremented_external = [(current_cap, self.externals[res][0] + atom_count)]
            self.bonds.extend(incremented_external)
            # For each tuple in self.bonds_dict[res] (list of bonds for that residue),
            # increment the tuples' values (the atoms' indices) by atom_count (the number of atoms up to that point)
            incremented_bonds = [tuple((group[0] + atom_count, group[1] + atom_count)) for group in self.bonds_dict[res]]
            self.bonds.extend(incremented_bonds)
            # Identify the externally bonding atom
            current_cap = self.externals[res][1] + atom_count
            # Increment the atom count for the next iteration in the loop
            atom_count += self.n_atoms[res]

        end_cap = self.order[-1]
        incremented_external = [(current_cap, self.externals[end_cap] + atom_count)]
        self.bonds.extend(incremented_external)
        incremented_bonds = [tuple((group[0] + atom_count, group[1] + atom_count)) for group in self.bonds_dict[end_cap]]
        self.bonds.extend(incremented_bonds)
        atom_count += self.n_atoms[end_cap]

        # Draw molecule as graph:
        for bond in self.bonds:
            a, b = bond
            # Clumsily convert from index to count
            self.topology.add_edge(a + 1, b + 1)

        # Uncomment to draw nice graph of results.
        draw(self.topology, with_labels=True, font_weight='bold')
        plt.show()

    def atom_types(self):
        """"""

        def gen_type_dict():
            for pos, atom in enumerate(self.molecule):
                # Add padding
                type_list = [atom[0], f'QUBE{str(200 + pos).zfill(4)}', f'{atom[0]}{str(200 + pos).zfill(4)}']
                yield pos, type_list

        self.AtomTypes = {atom_index: info_list for atom_index, info_list in gen_type_dict()}
