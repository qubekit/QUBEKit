#!/usr/bin/env python

import os
from QUBEKit import bonds, charges

import subprocess

# opt_molecule = bonds.read_pdb('methane.pdb')
#
# bonds.input_psi4(input_file='methane.pdb', opt_molecule=opt_molecule, charge=0, multiplicity=1, basis='6-311++G(d,p)',
#                  theory='wB97X-D', memory=2, input_option='c')
#
# subprocess.call('psi4 {}_freq.dat -n {}'.format('methane', 2), shell=True)
#
# charges.charge_gen('Dt.cube', 6, '/home/b8009890/Programs/chargemol_09_26_2017')
#
# subprocess.call('cp /home/b8009890/Programs/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_serial .', shell=True)
# subprocess.call('./Chargemol_09_26_2017_linux_serial job_control.txt', shell=True)

bonds.config_loader()
