#!/usr/bin/env python


class Ligand:

    def __init__(self, filename, smilesstring=None):

        self.filename = filename
        self.name = filename[:-4]
        self.molecule = None
        self.topology = None
        self.smiles = smilesstring
        self.angles = None
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
        self.descriptors = {}
        self.scan_energy = {}
        self.AtomTypes = {}
        self.Residues = {}
        self.HarmonicBondForce = {}
        self.HarmonicAngleForce = {}
        self.PeriodicTorsionForce = {}
        self.NonbondedForce = {}
        self.read_pdb()
        self.find_angles()
        self.find_dihedrals()
        self.find_rotatable_dihedrals()
        self.get_dihedral_values()
        self.get_bond_lengths()
        self.get_angle_values()

    element_dict = {'H': 1.008000,  # Group 1
                    'C': 12.011000,  # Group 4
                    'N': 14.007000, 'P': 30.973762,  # Group 5
                    'O': 15.999000, 'S': 32.060000,  # Group 6
                    'F': 18.998403, 'Cl': 35.450000, 'Br': 79.904000, 'I': 126.904470  # Group 7
                    }

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def read_pdb(self, QM=False, MM=False):
        """Reads the input PDB file to find the ATOM or HETATM tags, extracts the elements and xyz coordinates.
        Then reads through the connection tags and builds a connectivity network (only works if connections present in PDB file).
        Bonds are easily found through the edges of the network.
        Can also generate a simple plot of the network.
        """

        from re import sub
        from networkx import Graph, draw
        # import matplotlib.pyplot as plt

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

        from networkx import neighbors

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

        from networkx import neighbors

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
        """Take the topology graph network and dihedrals dictionary and for each dihedral in there work out if the torsion is
        rotatable. Returns a list of dihedral dictionary keys representing the rotatable dihedrals.
        """

        from networkx import has_path

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
        """Taking the molecules xyz coordinates and dihedrals dictionary the function returns a dictionary of dihedral
        angle keys and values. There is also the option to supply just the keys of the dihedrals you want to calculate.
        """

        from numpy import array, linalg, dot, degrees, cross, arctan2

        self.dih_phis = {}
        # Check if a rotatable tuple list is supplied, else calculate the angles for all dihedrals in the molecule.
        if self.rotatable:
            keys = self.rotatable

        else:
            keys = list(self.dihedrals.keys())

        if QM:
            molecule = self.QMoptimized

        elif MM:
            molecule = self.MMoptimized

        else:
            molecule = self.molecule

        for key in keys:
            torsions = self.dihedrals[key]
            for torsion in torsions:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1 = array(molecule[int(torsion[0])-1][1:])
                x2 = array(molecule[int(torsion[1])-1][1:])
                x3 = array(molecule[int(torsion[2])-1][1:])
                x4 = array(molecule[int(torsion[3])-1][1:])
                b1 = x2 - x1
                b2 = x3 - x2
                b3 = x4 - x3
                t1 = linalg.norm(b2) * dot(b1, cross(b2, b3))
                t2 = dot(cross(b1, b2), cross(b2, b3))
                dih = arctan2(t1, t2)
                dih = degrees(dih)
                self.dih_phis[torsion] = dih

        return self.dih_phis

    def get_bond_lengths(self, QM=False, MM=False):
        """For the given molecule and topology find the length of all of the bonds."""

        from numpy import array, linalg

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
        """For the given molecule and list of angle terms measure the angle values
        return a dictionary of angles and values.
        """

        from numpy import array, linalg, dot, arccos, degrees

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
            b1 = x1 - x2
            b2 = x3 - x2
            cosine_angle = dot(b1, b2) / (linalg.norm(b1) * linalg.norm(b2))
            theta = degrees(arccos(cosine_angle))
            self.angle_values[angle] = theta

        return self.angle_values

    def write_pdb(self, QM=False, MM=False):
        """Take the current molecule and topology and write a pdb file for the molecule."""
        pass

    def write_parameters(self):
        """Take the molecules parameter set and write an xml file for the molecule."""

        # first build the xml tree
        self.build_tree()

        import xml.etree.ElementTree as ET
        import xml.dom.minidom

        tree = self.xml_tree.getroot()
        messy = ET.tostring(tree, 'utf-8')

        xml = xml.dom.minidom.parseString(messy)
        pretty_xml_as_string = xml.toprettyxml(indent="")

        with open(f'{self.name}.xml', 'w+') as xml_doc:
            xml_doc.write(pretty_xml_as_string)

        print('XML made!')

    def build_tree(self):
        """This method seperates the parameters and builds an xml tree ready for writing out."""

        import xml.etree.ElementTree as ET

        # create XML layout
        root = ET.Element('ForceField')
        AtomTypes = ET.SubElement(root, "AtomTypes")
        Residues = ET.SubElement(root, "Residues")
        Residue = ET.SubElement(Residues, "Residue", name="UNK")
        HarmonicBondForce = ET.SubElement(root, "HarmonicBondForce")
        HarmonicAngleForce = ET.SubElement(root, "HarmonicAngleForce")
        PeriodicTorsionForce = ET.SubElement(root, "PeriodicTorsionForce")
        NonbondedForce = ET.SubElement(root, "NonbondedForce", attrib={'coulomb14scale': "0.5", 'lj14scale': "0.5"})

        # Add the AtomTypes
        for i in range(len(self.AtomTypes)):
            ET.SubElement(AtomTypes, "Type", attrib={'name': self.AtomTypes[i][1], 'class': self.AtomTypes[i][2],
                                                     'element': self.molecule[i][0],
                                                     'mass': str(self.element_dict[self.molecule[i][0]])})
            ET.SubElement(Residue, "Atom", attrib={'name': self.AtomTypes[i][0], 'type': self.AtomTypes[i][1]})

        # add the bonds/connections
        for key in self.HarmonicBondForce.keys():
            ET.SubElement(Residue, "Bond", attrib={'from': str(key[0]), 'to': str(key[1])})
            ET.SubElement(HarmonicBondForce, "Bond",
                          attrib={'class1': self.AtomTypes[key[0]][2], 'class2': self.AtomTypes[key[1]][2],
                                  'length': self.HarmonicBondForce[key][0], 'k': self.HarmonicBondForce[key][1]})

        # add the angles
        for key in self.HarmonicAngleForce.keys():
            ET.SubElement(HarmonicAngleForce, "Angle",
                          attrib={'class1': self.AtomTypes[key[0]][2], 'class2': self.AtomTypes[key[1]][2],
                                  'class3': self.AtomTypes[key[2]][2], 'angle': self.HarmonicAngleForce[key][0],
                                  'k': self.HarmonicAngleForce[key][1]})

        # add the torsion terms
        for key in self.PeriodicTorsionForce.keys():
            ET.SubElement(PeriodicTorsionForce, "Proper",
                          attrib={'class1': self.AtomTypes[key[0]][2], 'class2': self.AtomTypes[key[1]][2],
                                  'class3': self.AtomTypes[key[2]][2], 'class4': self.AtomTypes[key[3]][2],
                                  'k1': self.PeriodicTorsionForce[key][0][1], 'k2': self.PeriodicTorsionForce[key][1][1],
                                  'k3': self.PeriodicTorsionForce[key][2][1], 'k4': self.PeriodicTorsionForce[key][3][1],
                                  'periodicity1': '1', 'periodicity2': '2', 'periodicity3': '3', 'periodicity4': '4',
                                  'phase1': self.PeriodicTorsionForce[key][0][2], 'phase2': self.PeriodicTorsionForce[key][1][2],
                                  'phase3': self.PeriodicTorsionForce[key][2][2], 'phase4': self.PeriodicTorsionForce[key][3][2]})

        # add the non-bonded parameters
        for key in self.NonbondedForce.keys():
            ET.SubElement(NonbondedForce, "Atom",
                          attrib={'type': self.AtomTypes[key][1], 'charge': self.NonbondedForce[key][0], 'sigma': self.NonbondedForce[key][1], 'epsilon': self.NonbondedForce[key][2]})

        # Store the tree back into the molecule
        tree = ET.ElementTree(root)
        self.xml_tree = tree

    def read_xyz_geo(self):
        """Read a geometric opt.xyz file to find the molecule array structure."""

        opt_molecule = []
        write = False

        # opt.xyz is the geometric optimised structure file.
        with open('opt.xyz', 'r') as opt:
            lines = opt.readlines()
            for line in lines:
                if 'Iteration' in line:
                    print(f'Optimisation converged at iteration {int(line.split()[1])} with final energy {float(line.split()[3])}')
                    write = True

                elif write:
                    opt_molecule.append([line.split()[0], float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
        self.QMoptimized = opt_molecule
        return self

    def read_xyz(self, QM=False, MM=True):
        """Read a general xyz file format and return the structure array.
        QM and MM decide where it will be stored in the molecule.
        """
        pass
