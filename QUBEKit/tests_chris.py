#! /usr/bin/env python

from QUBEKit.helpers import Configure
from QUBEKit.engines import QCEngine
from QUBEKit.ligand import Ligand

from collections import OrderedDict

defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


# mol = Ligand('methane.pdb')
# qcengine = QCEngine(mol, config_dict)
#
# ret = qcengine.call_qcengine('geo', 'gradient', 'input')
#
# # Working keys for a geometric test
# # print(ret.energies)
# # print(ret.trajectory)
# # print(ret.final_molecule)
#
# print(ret.energies)

a = OrderedDict([('parametrise', 2436), ('mm_optimise', 4567),
                 ('qm_optimise', 78), ('hessian', 4356)])

print(list(a).index('mm_optimise'))
print(list(a).index('hessian'))
