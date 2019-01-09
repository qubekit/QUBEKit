#! /usr/bin/env python

from QUBEKit.engines import PSI4, Gaussian, Chargemol
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan
from QUBEKit.lennard_jones import LennardJones as LJ
from QUBEKit.modseminario import ModSeminario
from QUBEKit import smiles, decorators
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, pretty_progress, pretty_print

import os
from subprocess import call as sub_call

# coords = [[0.345, 1.456, 2.456], [1.345, 4.345, 8.345], [9.435, 5.234, 2.456]]
#
# print(coords)
#
# for coord in coords:
#
#     coord.sort()
#
# print(coords)
