#!/usr/bin/env python

from QUBEKit.ligand import Protein
from QUBEKit.parametrisation import XMLProtein
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.proteinTools import qube_general, pdb_reformat, get_water
from os import remove
from sys import argv


def main():
    """This script is used to prepare proteins with the QUBE FF
    1) prepare the protein for onetep using --setup which prints an xyz of the system
    2) after the onetep calculation bring back the ddec.onetep file and prameterize the system with --build
     this must be the same pdb used in setup as the atom order must be retained."""

    # start reading the command line
    commands = argv[1:]

    for count, cmd in enumerate(commands):

        # first case is setup
        # read the pdb file, check the amount of residues and make an xyz file for onetep
        if cmd == '--setup':
            pdb_file = commands[count + 1]
            print(pdb_file)
            print('starting protein prep, reading pdb file...')
            protein = Protein(pdb_file)
            print(f'{len(protein.Residues)} residues found!')
            # TODO find the magic numbers for the box for onetep
            protein.write_xyz(name='protein')
            print(f'protein.xyz file made for ONETEP\n Run this file')

        # here we use the results of the onetep calculation and the general QUBE_general.xml to make the custom system
        # and write a new pdb file with all of the atoms renamed and residues
        elif cmd == '--build':
            pdb_file = commands[count + 1]
            print(pdb_file)
            protein = Protein(pdb_file)
            # print the qube general FF to use in the parameterization
            qube_general()
            # now we want to add the connections and parametrise the protein
            XMLProtein(protein)
            # print(protein.HarmonicBondForce)
            # this updates the bonded info that is now in the object

            # finally we need the non-bonded parameters from onetep
            # fake configs as this will always be true

            configs = [{'charge': 0}, {'charges_engine': 'onetep'}, {}, {}]
            lj = LennardJones(protein, config_dict=configs)
            protein.NonbondedForce = lj.calculate_non_bonded_force()
            # now we write out the final parameters
            # we should also calculate the charges and lj at this point!
            print('Writing pdb file with conections...')
            protein.write_pdb(name='QUBE_pro')
            print('Writing XML file for the system...')
            protein.write_parameters(name='QUBE_pro', protein=True)
            # now remove the qube general file
            remove('QUBE_general_pi.xml')
            print('Done')

        # if we request a water model as well print it here.
        elif cmd == '--water':
            water = commands[count + 1]
            get_water(water)

        # after running a simulation in OpenMM we need to back convert the atom and residue names so we can post process
        elif cmd == '--convert':
            target = commands[count + 1]
            refernece = commands[count + 2]
            print(f'Rewriting input:{target} to match:{refernece}...')
            pdb_reformat(refernece, target)
            print('Done output made: QUBE_traj.pdb')

if __name__ == '__main__':
    main()