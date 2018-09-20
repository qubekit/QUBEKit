#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==================================================================================================
#|      _____                ____               __  __             __                             |
#|     /\  __`\             /\  _`\            /\ \/\ \     __    /\ \__                          |
#|     \ \ \/\ \    __  __  \ \ \L\ \     __   \ \ \/'/'   /\_\   \ \ ,_\                         |
#|      \ \ \ \ \  /\ \/\ \  \ \  _ <'  /'__`\  \ \ , <    \/\ \   \ \ \/                         |
#|       \ \ \\'\\ \ \ \_\ \  \ \ \L\ \/\  __/   \ \ \\`\   \ \ \   \ \ \_                        |
#|        \ \___\_\ \ \____/   \ \____/\ \____\   \ \_\ \_\  \ \_\   \ \__\                       | 
#|         \/__//_/  \/___/     \/___/  \/____/    \/_/\/_/   \/_/    \/__/                       |
#==================================================================================================
#|                                 Quantum Bespoke-kit                                            |
#==================================================================================================
#Utility for the derivation of specific ligand parameters

import argparse, sys

#Main function parser
def main():
    parser = argparse.ArgumentParser(prog='QUBEKit', formatter_class=argparse.RawDescriptionHelpFormatter,
description="""
Utility for the derivation of specific ligand parameters
Requires BOSS make sure BOSSdir is set in bashrc
Example input to write the bond and angles input file for g09 from a zmat file
python QUBEKit.py -f bonds -t write -z toluene.z -c 0 -m 1
File names NB/NBV non-bonded with/out virtual sites, BA bonds and angles fited, D dihedrals fitted
Final file name QuBe will be wrote when xml and GMX files are made
""")
    parser.add_argument('-f', "--function", help='Enter the function you wish to use bonds (covers bonds and angles terms), dihedrals and charges etc') 
    parser.add_argument('-t', "--type", help='Enter the function type you want this can be write , fit or analyse in the case of dihedrals (xyz charge input is wrote when bonds are fit) when writing and fitting dihedrals you will be promted for the zmat index scanning')
    parser.add_argument('-X', "--XML", help='Option for making a XML file  and GMX gro and itp files if needed options yes default is no')
    parser.add_argument('-z', '--zmat', help='The name of the zmat with .z')
    parser.add_argument('-p', '--PDB', help='The name of the pdb file with .pdb')
    parser.add_argument('-c', '--charge', default=0, help='The charge of the molecule nedded for the g09 input files, defulat = 0')
    parser.add_argument('-m', '--multiplicity', default=1, help='The multiplicity of the molecule nedded for the g09 input files, defult = 1')
    parser.add_argument('-s', '--submission', help='Write a sbmission script for the function called default no ')
    parser.add_argument('-v', '--Vn', help='The amount of Vn coefficients to fit in the dihedral fitting, default = 4 ')
    parser.add_argument('-l', '--penalty', help='The penalty used in torsion fitting between new parameters and opls reference, default = 0')
    parser.add_argument('-d', '--dihedral', help='Enter the dihedral number to be fit, default will look at what SCAN folder it is running in')
    parser.add_argument('-FR', '--frequency', help='Option to perform a QM MM frequency comparison, options yes default no')
    parser.add_argument('-SP', '--singlepoint', help='Option to perform a single point energy calculation in openMM to make sure the eneries match (OPLS combination rule is used)')
    parser.add_argument('-r', '--replace', help='Option to replace any valid dihedral terms in a molecule with QuBeKit previously optimised values. These dihedrals will be ignored in subsequent optimizations')
    parser.add_argument('-con', '--config', nargs='+', help='''Update global default options
qm: theory, vib_scaling, processors, memory.
fitting: dihstart, increment, numscan, T_weight, new_dihnum, Q_file, tor_limit, div_index
example: QuBeKit.py -con qm.theory wB97XD/6-311++G(d,p)''')
    parser.add_argument('-e', '--engine', default="psi4", choices=["psi4","g09"], help='select the qm engine used for optimisation calculations')
    args = parser.parse_args()
    if not args.zmat and not args.PDB and args.type!='analyse':
       sys.exit('Zmat or PDB missing please enter')
    elif args.function == 'bonds' and args.type == 'write' and args.PDB:
       import bonds
       #test if psi4 is avilable
       try:
           import psi4
           print('psi4 imported!')
       except:
           sys.exit('ImportError psi4 not found!')
       #pull in the config settings needed to use QM software
       theory, basis, vib_scaling, processors, memory, dihstart, increment, numscan, T_weight, new_dihnum, Q_file,tor_limit, div_index = bonds.config_setup()
       molecule_name = args.PDB[:-4]
       #convert the ligpargen pdb to psi4 format
       bonds.pdb_to_psi4(args.PDB, args.charge, args.multiplicity, basis, theory) 
       #Now run the optimiziation in psi4 using geometric
       
        


       
if __name__ == '__main__':
   main()
