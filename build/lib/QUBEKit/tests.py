#!/usr/bin/env python

# import os
import bonds

import subprocess

# opt_molecule = bonds.read_pdb('methane.pdb')

# test_file = bonds.input_psi4(input_file='methane.pdb', opt_molecule=opt_molecule, charge=0, multiplicity=1, basis='6-311G', theory='wb97x-d', memory=2, input_option='c')

# subprocess.call('psi4 %s_freq.dat -n %i' % ('methane', 2), shell=True)

theory, basis, vib_scaling, processors, memory, dihstart, increment, numscan, T_weight, new_dihnum, Q_file, tor_limit, div_index, chargemol = bonds.config_setup()
print(chargemol)


# charges.charge_gen('Dt.cube', 6, path)

