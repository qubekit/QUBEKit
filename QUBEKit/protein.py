from itertools import groupby, chain
from re import sub
from networkx import Graph, draw
import matplotlib.pyplot as plt


class Protein:
    """Class to handle proteins as chains of amino acids.
    Dicts are used to provide any standard parameters;
    the orders of the residues and atoms in the residues is constructed from the pdb file provided.
    """

    def __init__(self, filename):
        """
        filename                str; full name of the file e.g. haem.pdb
        name                    str; filename without the extension e.g. haem
        molecule                list of lists; Inner list is the atom type (str) followed by its coords (float)
                                e.g. [['C', -0.022, 0.003, 0.017], ['H', -0.669, 0.889, -0.101], ...]
        atom_names              list of str;
        order                   list of str; list of the residues in the protein, in order.
        bonds                   list of tuples of two ints; describes each bond in the molecule where the ints are the atom indices.
        bonds_dict              dict, key=str, val=list of tuples; key=residue name: val=list of the bond connections which are tuples
                                e.g. {'leu': [(0, 1), (1, 2), ...], ...}
        externals               dict, key=str, val=tuple, int or None; Similar to bonds_dict except this is only for the external bonds.
                                Some keys will have two external bonds, caps only have one, and others (e.g. ions) don't have any.
        bond_constants
        angles_dict
        propers_dict
        impropers_dict

        """

        self.filename = filename
        self.name = filename[:-4]
        self.molecule = []
        self.atom_names = []
        self.order = []
        self.bonds = []

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

    def read_pdb(self):
        """From the provided pdb file, extract the molecule coordinates, residue order and atom names."""

        with open(self.filename, 'r') as pdb:
            lines = pdb.readlines()

        all_residues = []
        molecule = []

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                element = str(line[76:78])
                element = sub('[0-9]+', '', element)
                element = element.replace(" ", "")
                self.atom_names.append(str(line.split()[2]))
                all_residues.append(line.split()[3].lower())

                # If the element column is missing from the pdb, extract the element from the name.
                if not element:
                    element = str(line.split()[2])[:-1]
                    element = sub('[0-9]+', '', element)

                molecule.append([element, float(line[30:38]), float(line[38:46]), float(line[46:54])])

        self.molecule = molecule
        self.order = [res for res, group in groupby(all_residues)]

    def identify_bonds(self):
        """Stitch together the residues' parameters from the xml file.

        For each residue add the external bond;
        this will be tuple[1] for the first residue and the tuple[0] for the next residue.
        Then add the internal bonds.
        All bonds will be incremented by however many atoms came before them in the chain.
        """

        # Append the cap's bonds to the bonds list
        self.bonds.extend(self.bonds_dict[self.order[0]])
        # Identify the current capping atom (the atom where the cap attaches to the protein)
        current_cap = self.externals[self.order[0]]
        # atom_count isn't always just the current_cap; sometimes the highest index atom is on a sidechain
        atom_count = self.n_atoms[self.order[0]]

        # Build topology to view molecule
        topology = Graph()

        # Start main loop, checking through all the chained residues (not the caps)
        for res in self.order[1:-1]:
            # Construct the tuple for the external bond
            external = [(current_cap, self.externals[res][0] + atom_count)]
            self.bonds.extend(external)
            # For each tuple in self.bonds_dict[res] (list of bonds for that residue),
            # increment the tuples' values (the atoms' indices) by atom_count (the number of atoms up to that point)
            incremented_bonds = [tuple((group[0] + atom_count, group[1] + atom_count)) for group in self.bonds_dict[res]]
            self.bonds.extend(incremented_bonds)
            # Identify the externally bonding atom
            current_cap = self.externals[res][1] + atom_count
            # Increment the atom count for the next iteration in the loop
            atom_count += self.n_atoms[res]

        end_cap = self.order[-1]
        external = [(current_cap, self.externals[end_cap] + atom_count)]
        self.bonds.extend(external)
        incremented_bonds = [tuple((group[0] + atom_count, group[1] + atom_count)) for group in self.bonds_dict[end_cap]]
        self.bonds.extend(incremented_bonds)
        atom_count += self.n_atoms[end_cap]

        # Draw molecule as graph:
        topology.add_node(i for i in self.molecule)
        for t in self.bonds:
            a, b = t
            a = self.atom_names[a]
            b = self.atom_names[b]
            topology.add_edge(a, b)

        draw(topology, with_labels=True, font_weight='bold')
        plt.show()