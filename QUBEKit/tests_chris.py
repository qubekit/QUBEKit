#! /usr/bin/env python

from QUBEKit.helpers import Configure
from QUBEKit.engines import QCEngine
from QUBEKit.ligand import Ligand

import qcengine as qcng
import qcelemental as qcel

defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


#
# mol = qcel.models.Molecule.from_data("""
# 0 1
# O  0.0  0.000  -0.129
# H  0.0 -1.494  1.027
# H  0.0  1.494  1.027
# """)
#
#
# inp = qcel.models.ResultInput(molecule=mol, driver='gradient', model={'method': 'SCF', 'basis': 'sto-3g'}, keywords={'scf_type': 'df'})
#
# ret = qcng.compute(inp, 'psi4')
#
# print(ret.return_result)
# print(ret.properties.scf_dipole_moment)


mol = Ligand('methane.pdb')
qcengine = QCEngine(mol, config_dict)

print(qcengine.call_qcengine('psi4', 'properties', 'input').return_result)
