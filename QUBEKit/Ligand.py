#!/usr/bin/env python

class Ligand(object):

    def __init__(self, filename, smilesstring=None):
        self.name = filename
        self.molecule = None
        self.topology = None
        self.smiles = smilesstring
        self.angles = None
        self.dihedrals = None
        self.rotatable = None
        self.dih_phis = None
        self.bond_lengths = None
        self.angle_values = None
        self.bonds = None
        self.MMoptimized = None
        self.QMoptimized = None
        self.parameters = None
        self.parameter_engine = None
        self.hessian = None
        self.read_pdb()
        self.find_angles()
        self.find_dihedrals()
        self.find_rotatable_dihedrals()
        self.get_dihedral_values()
        self.get_bond_lengths()
        self.get_angle_values()

    def __repr__(self):
        return "Molcule object infomation\n molecule name: {}\n molecule array : {}\n MMoptimized structure array : {}\n QMoptimized structure array : {} \n molecule topology : [nodes : {}, edges : {}]\n" \
               " molecule smiles : {}\n bonds : {}\n angles : {}\n dihedrals : {}\n rotatable dihedrals : {}\n" \
               " dihedral angles : {}\n bond lengths : {}\n measured angles : {}\n molecule parameters : {}\n parameter engine used : {}\n".format(self.name, self.molecule, self.MMoptimized, self.QMoptimized, self.topology.nodes,
               self.topology.edges, self.smiles, self.topology.edges, self.angles, self.dihedrals, self.rotatable, self.dih_phis, self.bond_lengths, self.angle_values, self.parameters, self.parameter_engine )

    def read_pdb(self):
        """Reads the input PDB file to find the ATOM or HETATM tags, extracts the elements and xyz coordinates.
        Then read through the connection tags and build connectivity network only works if connections present in PDB file.
        Bonds are easily found through the edges of the network.
        Can also generate a simple plot of the network."""

        from re import sub
        from networkx import Graph, draw
        #import matplotlib.pyplot as plt


        with open(self.name, 'r') as pdb:
            lines = pdb.readlines()

        self.molecule = []
        self.topology = Graph()

        j = 1  # atom counter used for graph node generation
        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                element = str(line[76:78])
                element = sub('[0-9]+', '', element)
                element = element.replace(" ", "")

                # If the element column is missing from the pdb, extract the element from the name.
                if  len(element) == 0:
                    element = str(line.split()[2])[:-1]
                    element = sub('[0-9]+', '', element)

                self.topology.add_node(j) # Also add the atom number as the node in the graph
                j += 1
                self.molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54])])

            if 'CONECT' in line: # Now look through the conectivitty section and add all edges to the graph coresponding to the bonds.
                for i in range(2,len(line.split())):
                    if int(line.split()[i]) != 0:
                        self.topology.add_edge(int(line.split()[1]), int(line.split()[i]))

        #draw(topology, with_labels=True, font_weight='bold') # uncomment these lines to draw the graph network generated from the pdb.
        #plt.show()

        return self.molecule, self.topology



    def find_angles(self):
        """Take the topology graph network and return a list of all angle combinations.
        Checked against OPLS-AA on molecules containing 10-63 angles."""

        from networkx import Graph, neighbors

        self.angles = []

        for node in self.topology.nodes:
            bonded = sorted(list(neighbors(self.topology, node)))
            if len(bonded) < 2:         # Check that the atom has more than one bond
                continue

            # Find all possible angle combinations from the list
            for i in range(len(bonded)):
                for j in range(i+1, len(bonded)):
                    atom1, atom3 = bonded[i], bonded[j]

                    self.angles.append((atom1, node, atom3))

        return self.angles



    def find_dihedrals(self):
        """Take the topology graph network and again return a dictionary of all possible dihedral combinations stored under
        the central bond keys which describe the angle."""

        from networkx import Graph, neighbors

        self.dihedrals = {}

        for edge in self.topology.edges: # Work through the network using each edge as a central dihedral bond

            for start in list(neighbors(self.topology, edge[0])):

                if start != edge[0] and start != edge[1]: # Check atom not in main bond

                    for end in list(neighbors(self.topology, edge[1])):

                        if end != edge[0] and end != edge[1]: # Check atom not in main bond

                            if edge not in self.dihedrals.keys():
                                self.dihedrals[edge] = [(start, edge[0], edge[1], end)] # Add the central edge as a key the first time it is used

                            else:

                                self.dihedrals[edge].append((start, edge[0], edge[1], end)) # Add the tuple to the correct key.

                        else:

                            continue

        return self.dihedrals




    def find_rotatable_dihedrals(self):
        """Take the topology graph network and dihedrals dictionary and for each dihedral in there work out if the torsion is
        rotatable. Returns a list of dihedral dictionary keys representing the rotatable dihedrals."""

        from networkx import Graph, has_path

        self.rotatable = []

        for key in self.dihedrals.keys(): # For each dihedral key remove the edge from the network
            self.topology.remove_edge(*key)

            if has_path(self.topology, key[0], key[1]): # Check if their is still a path between the two atoms in the edges.
                pass

            else:

                self.rotatable.append(key)

            self.topology.add_edge(*key) # Add edge back to the network and try next key

        return self.rotatable


    def get_dihedral_values(self):
        """Taking the molecules xyz coordinates and dihedrals dictionary the function returns a dictionary of dihedral
        angle keys and values. There is also the option to supply just the keys of the dihedrals you want to calculate."""

        from numpy import array, linalg, dot, degrees, cross, arctan2

        self.dih_phis = {}
        # Check to see if a ratable tuple list is supplied else calculate the angles for all dihedrals in the molecule.
        if self.rotatable:
            keys = self.rotatable

        else:
           keys = list(self.dihedrals.keys())

        for key in keys:
            torsions = self.dihedrals[key]
            for torsion in torsions:
                # Calculate the dihedral angle in the molecule using the molecule data array.
                x1 = array(self.molecule[int(torsion[0])-1][1:])
                x2 = array(self.molecule[int(torsion[1])-1][1:])
                x3 = array(self.molecule[int(torsion[2])-1][1:])
                x4 = array(self.molecule[int(torsion[3])-1][1:])
                b1 = x2 - x1
                b2 = x3 - x2
                b3 = x4 - x3
                t1 = linalg.norm(b2) * dot(b1, cross(b2, b3))
                t2 = dot(cross(b1, b2), cross(b2, b3))
                dih = arctan2(t1, t2)
                dih = degrees(dih)
                self.dih_phis[torsion] = dih

        return self.dih_phis


    def get_bond_lengths(self):
        """For the given molecule and topology find the length of all of the bonds."""

        from numpy import array, linalg

        self.bond_lengths = {}


        for edge in self.topology.edges:
            atom1 = array(self.molecule[int(edge[0])-1][1:])
            atom2 = array(self.molecule[int(edge[1])-1][1:])
            bond_dist = linalg.norm(atom2 - atom1)
            self.bond_lengths[edge] = bond_dist

        return self.bond_lengths



    def get_angle_values(self):
        """For the given molecule and list of angle terms measure the angle values
        return a dictionary of angles and values."""

        from numpy import array, linalg, dot, arccos, degrees
        self.angle_values = {}

        for angle in self.angles:
            x1 = array(self.molecule[int(angle[0])-1][1:])
            x2 = array(self.molecule[int(angle[1])-1][1:])
            x3 = array(self.molecule[int(angle[2])-1][1:])
            b1 = x1 - x2
            b2 = x3 - x2
            cosine_angle = dot(b1, b2) / (linalg.norm(b1) * linalg.norm(b2))
            theta = degrees(arccos(cosine_angle))
            self.angle_values[angle] = theta

        return self.angle_values

    def write_PDB(self):
        """Take the current molecule and topology and write a pdb file for the molecule."""
        pass

    def write_parameters(self):
        """Take the molecules parameter set and write an xml file for the molecule."""
        pass