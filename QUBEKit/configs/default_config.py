#!/usr/bin/env python

# QuBeKit config file allows users to reset the global variables


qm = {'theory': 'BB1K', 'basis': '6-311++G(d,p)', 'vib_scaling': 0.957, 'threads': 2, 'memory': 2}

# qm configs:

# theory: g09 theory to use in freq and dihedral scans recommended wB97XD/6-311++G(d,p)

# basis:

# vib_scaling: the associated scaling to the theory

# threads: the number of processors to be used in the g09 scripts this affects the bonds and dihedral scans

# memory: the amount of memory (in GB) to be specified in the g09 scripts


fitting = {'dih_start': 0, 'increment': 15, 'num_scan': 25, 't_weight': 'infinity', 'new_dih_num': 501, 'q_file': 'results.dat', 'tor_limit': 20, 'div_index': 0}

# fitting configs:

# dih_start: starting angle of dihedral scan

# increment: angle increase increment

# num_scan: the number of optimisations around the dihedral angle

# t_weight: the weighting temperature that can be changed to better fit complicated surfaces

# new_dih_num: the parameter number for the new dihedral to be fit

# q_file: if the results are collected with QuBeKit this is always true

# tor_limit: torsion Vn limit to speed up fitting

# div_index: fitting starting index in the division array


paths = {'chargemol': '/home/b8009890/Programs/chargemol_09_26_2017'}

# path configs:

# chargemol: location of the chargemol program directory
