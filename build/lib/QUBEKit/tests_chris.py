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


file = 'methane.pdb'
mol = Ligand(file)

qm_engine = Gaussian(mol, get_mol_data_from_csv()['default'])

# qm_engine.generate_input(optimize=True, hessian=True)

qm_engine.hessian()
