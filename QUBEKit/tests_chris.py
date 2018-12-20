#! /usr/bin/env python

from QUBEKit.engines import PSI4, Gaussian, Chargemol
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan
from QUBEKit.lennard_jones import LennardJones as LJ
from QUBEKit.modseminario import ModSeminario
from QUBEKit import smiles, decorators
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, pretty_progress

import os
from subprocess import call as sub_call


# file = 'methanol.pdb'
# mol = Ligand(file)
#
# pretty_progress()

# print('{:15} {:15} {:15}'.format('Name', 'Parametrised', 'Optimised'))
# print('{:15} {:15}'.format('Chris', '1'))
# print('{:15} {:15}'.format('Christopher', '1'))
# print("\033[1;32;40 m Bright Green  \n")
# print("\x1b[1;31;40m {:>13d}{:>13d}".format(1, 2))
# print('\033[1;33mHello \033[0;0mworld')


a = {'PBE': 'PBEPBE'}
b = 'PBE'
