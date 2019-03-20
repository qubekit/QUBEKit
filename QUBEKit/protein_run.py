#!/usr/bin/env python

from QUBEKit.ligand import Protein
from QUBEKit.parametrisation import XMLProtein
from QUBEKit.lennard_jones import LennardJones

from sys import argv


def main():

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
            print(f'{len(protein.residues)} residues found!')
            protein.write_xyz(name='protein')
            print(f'protein.xyz file made for ONETEP\n Run this file')

        # here we use the results of the onetep calculation and the general QUBE_general.xml to make the custom system
        # and write a new pdb file with all of the atoms renamed and residues
        elif cmd == '--build':
            pdb_file = commands[count + 1]
            print(pdb_file)
            protein = Protein(pdb_file)

            # now we want to add the connections and parametrise the protein
            XMLProtein(protein)
            # print(protein.HarmonicBondForce)
            # this updates the bonded info that is now in the object

            # finally we need the non-bonded parameters from onetep
            # fake configs as this will always be true
            configs = [{'charge': 0}, {'charges_engine': 'onetep'}, {}, {}]
            lj = LennardJones(protein, config_dict=configs)
            protein.NonbondedForce = lj.calculate_non_bonded_force()
            print(protein.NonbondedForce)
            # now we write out the final parameters
            # we should also calculate the charges and lj at this point!
            protein.write_pdb(name='QUBE_pro')
            protein.write_parameters(name='QUBE_pro', protein=True)
            print('reading file and adding connections')


if __name__ == '__main__':
    main()
