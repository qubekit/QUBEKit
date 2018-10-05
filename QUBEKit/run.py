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
# Utility for the derivation of specific ligand parameters

import argparse
import sys
import os
import subprocess


# TODO move old QUBEKit functions that support boss to function list.

# TODO Add intelligent order handling to future-proof any new functionality.
 

def main():
    """Main function parser. Reads commands from terminal and executes accordingly."""

    parser = argparse.ArgumentParser(prog='QUBEKit', formatter_class=argparse.RawDescriptionHelpFormatter,
description="""
Utility for the derivation of specific ligand parameters
Requires BOSS make sure BOSSdir is set in bashrc
Example input to write the bond and angles input file for g09 from a zmat file
python QUBEKit.py -f bonds -t write -z toluene.z -c 0 -m 1
File names NB/NBV non-bonded with/out virtual sites, BA bonds and angles fited, D dihedrals fitted
Final file name QuBe will be wrote when xml and GMX files are made
""")

    parser.add_argument('-f', '--function', help='Enter the function you wish to use bonds (covers bonds and angles terms), dihedrals and charges etc')
    parser.add_argument('-t', '--type', help='Enter the function type you want this can be write , fit or analyse in the case of dihedrals (xyz charge input is wrote when bonds are fit) when writing and fitting dihedrals you will be promted for the zmat index scanning')
    parser.add_argument('-X', '--XML', help='Option for making a XML file  and GMX gro and itp files if needed options yes default is no')
    parser.add_argument('-z', '--zmat', help='The name of the zmat with .z')
    parser.add_argument('-p', '--PDB', help='The name of the pdb file with .pdb')
    parser.add_argument('-c', '--charge', default=0, help='The charge of the molecule nedded for the g09 input files, defulat = 0')
    parser.add_argument('-m', '--multiplicity', default=1, help='The multiplicity of the molecule nedded for the g09 input files, defult = 1')
    parser.add_argument('-s', '--submission', help='Write a submission script for the function called default no ')
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
    parser.add_argument('-g', '--geometric', action='store_false', default=True, help='Use geometric/crank(torsiondrive) in optimisations?')
    parser.add_argument('-e', '--engine', default="psi4", choices=["psi4","g09"], help='Select the qm engine used for optimisation calculations')
    parser.add_argument('-sm', '--smiles', help='Enter the SMILES code for the molecule')
    args = parser.parse_args()

    # if not args.zmat and not args.PDB and not args.smiles and args.type != 'analyse':
    #     sys.exit('Zmat, PDB or smiles missing please enter')

    from QUBEKit import bonds

    # Load config dictionaries
    qm, fitting, paths = bonds.config_loader()

    if args.function == 'bonds' and args.type == 'write' and args.PDB:
        molecule_name = args.PDB[:-4]
        # Convert the ligpargen pdb to psi4 format
        molecule = bonds.read_pdb(args.PDB)

        if args.engine == 'psi4' and args.geometric:
            print('writing psi4 style input file for geometric')
            bonds.pdb_to_psi4_geo(args.PDB, molecule, args.charge, args.multiplicity, qm['basis'], qm['theory'])
            # Now run the optimisation in psi4 using geometric
            run = input('would you like to run the optimization? >')

            if run.lower() in ('y' or 'yes'):
                # Test if psi4 or g09 is available (add function to look for g09) search the environment list not import
                os.chdir('BONDS/')

                try:
                    print('calling geometric and psi4 to optimizie')
                    log = open('log.txt', 'w+')
                    subprocess.call('geometric-optimize --{} {}.psi4in --nt {}'.format(args.engine, molecule_name, qm['threads']), shell=True, stdout=log)
                    log.close()

                except:

                    # TODO Find a way of running psi4 from anywhere. Currently this only works inconsistently.

                    print('psi4 missing.')
                    sys.exit()

                # Make sure optimization has finished
                opt_molecule = bonds.get_molecule_from_psi4()

                # Optimised molecule structure stored in opt_molecule now prep psi4 file
                bonds.input_psi4(args.PDB, opt_molecule, args.charge, args.multiplicity, qm['basis'], qm['theory'], qm['memory'])

                print('calling psi4 to calculate frequencies and Hessian matrix')

                # Call psi4 to perform frequency calc
                subprocess.call('psi4 {}_freq.dat freq_out.dat -n {}'.format(molecule_name, qm['threads']), shell=True)

                # Now check and extract the formated hessian in N * N form
                form_hess = bonds.extract_hessian_psi4(opt_molecule)

                print('calling modified seminario method to calculate bonded force constants')
                print(form_hess)

                # import Modseminario
                # Modseminario.modified_Seminario_method(vib_scaling, form_hess)

        elif args.engine == 'psi4' and not args.geometric:
            print('writing psi4 style input file')
            bonds.pdb_to_psi4(args.PDB, molecule, args.charge, args.multiplicity, qm['basis'], qm['theory'], qm['memory'])

            # Now make new pdb from output xyz? Do we need it apart from for ONETEP?
            # Feed the xzy back to psi4 to run frequency calc and get hessian out

        elif args.engine == 'g09':
            print('writing g09 style input file')
            bonds.pdb_to_g09(args.PDB, molecule, args.charge, args.multiplicity, qm['basis'], qm['theory'], qm['threads'], qm['memory'])
            print('geometric does not support gaussian09 please optimise separately')

    elif args.function == 'bonds' and args.type == 'fit' and args.PDB and args.engine == 'psi4':

        from QUBEKit import bonds, Modseminario

        bonds.extract_hessian_psi4(molecule=bonds.read_pdb(args.PDB))

        print('calling the modified seminario method to calculate bonded terms from psi4 output')
        print('searching for hessian matrix')

        # Get optimized structure from psi4
        try:
            opt_molecule = bonds.get_molecule_from_psi4()
        except:
            try:
                os.chdir('BONDS/')
                opt_molecule = bonds.get_molecule_from_psi4()
            except:
                sys.exit('opt.xyz file not found!')

        try:
            form_hess = bonds.extract_hess_psi4(opt_molecule)
        except:
            try:
                os.chdir('BONDS/')
                print(os.getcwd())
                form_hess = bonds.extract_hess_psi4(opt_molecule)
            except:
                try:
                    os.chdir('FREQ/')
                    print(os.getcwd())
                    form_hess = bonds.extract_hess_psi4(opt_molecule)
                except:
                    sys.exit('hessian missing')

        # Needs a section to make sure there is a general parameter file
        Modseminario.modified_Seminario_method(qm['vib_scaling'], form_hess, args.engine, opt_molecule)

    # Smiles processing
    elif args.smiles:

        from QUBEKit import smiles

        smiles.smiles_to_pdb(args.smiles)


if __name__ == '__main__':
    main()
