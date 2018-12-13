#!/usr/bin/env python


# QuBeKit config file allows users to reset the global variables


qm = {
    'theory': 'B3LYP',              # g09 theory to use in freq and dihedral scans recommended wB97XD
    'basis': '6-311++G(d,p)',       # basis set
    'vib scaling': 0.957,           # associated scaling to the theory
    'threads': 6,                   # number of processors used in g09; affects the bonds and dihedral scans
    'memory': 2,                    # amount of memory (in GB); specified in the g09 scripts
    'convergence': 'GAU_TIGHT',     # criterion used during optimisations; works using psi4 and geometric so far
    'iterations': 100,              # max number of optimisation iterations
    'bonds engine': 'psi4',
    'charges engine': 'chargemol',
    'ddec version': 6,
    'geometric': True,
    'solvent': False,

}

fitting = {
    'dih_start': 0,                 # starting angle of dihedral scan
    'increment': 15,                # angle increase increment
    'num_scan': 25,                 # number of optimisations around the dihedral angle
    't_weight': 'infinity',         # weighting temperature that can be changed to better fit complicated surfaces
    'new_dih_num': 501,             # parameter number for the new dihedral to be fit
    'q_file': 'results.dat',        # if the results are collected with QuBeKit this is always true
    'tor_limit': 20,                # torsion Vn limit to speed up fitting
    'div_index': 0,                 # fitting starting index in the division array
    'parameter_enegine': 'openFF',  # method used for initial parametrisation
}

descriptions = {
    'chargemol': '/home/b8009890/Programs/chargemol_09_26_2017',    # location of the chargemol program directory
    'log': 999
}
