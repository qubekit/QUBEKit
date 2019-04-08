#! /usr/bin/env python

from QUBEKit.helpers import Configure

defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


a = 'dsfg'
b = 'sdfg'

if not (a and b):
    print('no')
else:
    print('yes')

