#! /usr/bin/env python

from QUBEKit.engines import PSI4, Gaussian, Chargemol, ONETEP
from QUBEKit.ligand import Ligand, Protein, Protein2
# from QUBEKit.dihedrals import TorsionScan
from QUBEKit.lennard_jones import LennardJones as LJ
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, pretty_progress, pretty_print, Configure
from QUBEKit.decorators import exception_logger_decorator
# from QUBEKit.parametrisation import Parametrisation, OpenFF, AnteChamber, XML
from QUBEKit import smiles


defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


def main():

    mol = Ligand('methane.pdb')

    onetep = ONETEP(mol, config_dict)

    onetep.calculate_hull()


if __name__ == '__main__':

    main()
