#!/usr/bin/env python


# QuBeKit config file allows users to reset the global variables


qm = {
    'theory': 'PBEPBE',             # Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example
    'basis': '6-311++G(d,p)',       # Basis set
    'vib_scaling': 0.991,           # Associated scaling to the theory
    'threads': 6,                   # Number of processors used in g09; affects the bonds and dihedral scans
    'memory': 2,                    # Amount of memory (in GB); specified in the g09 scripts
    'convergence': 'GAU_TIGHT',     # Criterion used during optimisations; works using psi4 and geometric so far
    'iterations': 100,              # Max number of optimisation iterations
    'bonds_engine': 'psi4',         # Engine used for bonds calculations
    'charges_engine': 'chargemol',  # Engine used for charges calculations
    'ddec_version': 6,              # DDEC version used by chargemol, 6 recommended but 3 is also available
    'geometric': True,              # Use geometric for optimised structure (if False, will just use psi4)
    'solvent': True,                # Use a solvent in the psi4/gaussian09 input
}

fitting = {
    'dih_start': 0,                 # Starting angle of dihedral scan
    'increment': 15,                # Angle increase increment
    'num_scan': 25,                 # Number of optimisations around the dihedral angle
    't_weight': 'infinity',         # Weighting temperature that can be changed to better fit complicated surfaces
    'new_dih_num': 501,             # Parameter number for the new dihedral to be fit
    'q_file': 'results.dat',        # If the results are collected with QuBeKit this is always true
    'tor_limit': 20,                # Torsion Vn limit to speed up fitting
    'div_index': 0,                 # Fitting starting index in the division array
    'parameter_engine': 'openff',   # Method used for initial parametrisation
}

descriptions = {
    'chargemol': '/home/b8009890/Programs/chargemol_09_26_2017',    # Location of the chargemol program directory
    'log': '999',                   # Default string for the working directories and logs
}
