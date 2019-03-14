#! /usr/bin/env python

from QUBEKit.engines import PSI4, Gaussian, Chargemol
from QUBEKit.ligand import Ligand
# from QUBEKit.dihedrals import TorsionScan
from QUBEKit.lennard_jones import LennardJones as LJ
from QUBEKit.modseminario import ModSeminario
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, pretty_progress, pretty_print, Configure
from QUBEKit.decorators import exception_logger_decorator
from QUBEKit.parametrisation import Parametrisation, OpenFF, AnteChamber, XML
# from QUBEKit import smiles
from QUBEKit.ligand import Protein

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from numpy import arange
from numpy.polynomial.polynomial import polyfit
from operator import itemgetter
from itertools import chain


# def main():
#
#     ddec_data = []
#
#     ddec_version = 3
#
#     if ddec_version == 6:
#         name = 'DDEC6_even_tempered_net_atomic_charges.xyz'
#     elif ddec_version == 3:
#         name = 'DDEC3_net_atomic_charges.xyz'
#     else:
#         raise Exception('Invalid ddec version.')
#
#     with open(name, 'r+') as charge_file:
#
#         lines = charge_file.readlines()
#
#         # Find number of atoms
#         atom_total = int(lines[0])
#
#         for count, row in enumerate(lines):
#
#             if 'The following XYZ' in row:
#
#                 start_pos = count + 2
#
#                 for line in lines[start_pos:start_pos + atom_total]:
#                     # Append the atom number and type, coords, charge, dipoles:
#                     # ['atom number', 'atom type', 'x', 'y', 'z', 'charge', 'x dipole', 'y dipole', 'z dipole']
#                     atom_string_list = line.split()
#                     # Append all the float values first.
#                     atom_data = atom_string_list[2:9]
#                     atom_data = [float(datum) for datum in atom_data]
#
#                     # Prepend the first two values (atom_type = str, atom_number = int)
#                     atom_data.insert(0, atom_string_list[1])
#                     atom_data.insert(0, int(atom_string_list[0]))
#
#                     ddec_data.append(atom_data)
#                 break
#
#     charges = [atom[5] for atom in ddec_data]
#     atoms = [atom[1] for atom in ddec_data]
#     for charge in charges:
#         print(charge)
#     for atom in atoms:
#         print(atom)


# def main():
#     """Currently you cannot pass marker style as a list in Matplotlib.
#     To get around that, the data could be grouped by atom type and the colour passed as a list.
#     Then it would be possible to plot atoms which are the same type together,
#     while also maintaining the correct colours from the molecule names.
#
#     Anyway, this function collects the data from the txt file for ONETEP, DDEC3&6 charges.
#     The data are stored in a dictionary where the keys are the molecule names
#     and the values are a list of lists. Each nested list is an atom in the form:
#     ['Atom Type', 'DDEC3 charge', 'DDEC6 charge', 'ONETEP charge']
#
#     Overall, this gives something like:
#
#     data = {'Benzene': [['C', '-0.108887', '-0.101384', '-0.1146'], ['C', ... ], ... ], ... }
#
#     The data are transformed to floats in the graphing section.
#     """
#
#     data = dict()
#
#     # Extract molecule data, using 'Atom' header as start marker, and 'p' as end marker
#     with open('comparison_data2.txt', 'r') as file:
#
#         lines = file.readlines()
#         for count, line in enumerate(lines):
#             if line.startswith('Atom'):
#                 data[lines[count - 1][:-1]] = []
#                 i = 1
#                 while lines[count + i].split()[0] != 'p':
#                     data[lines[count - 1][:-1]].append(lines[count + i].split())
#                     i += 1
#
#     # Can change colours used here if you want. Currently just using defaults.
#     colours = {'Acetamide': 'r', 'Benzene': 'b', 'Acetic acid': 'y',
#                'Acetophenone': 'k', 'Aniline': 'g', 'DMSO': 'm',
#                '2-Heptanone': 'c', '1-Octanol': 'grey', 'Phenol': 'indigo',
#                'Pyridine': 'olive'}
#
#     print(data)
#
#     # Marker dict to change element names to Matplotlib marker styles.
#     markers = {'O': 'o', 'C': '<', 'H': 's', 'N': 'd', 'S': '*'}
#
#     # Apply marker changes across all molecules' atoms (massive pain to implement into graph so currently useless).
#     for key, val in data.items():
#         for i in range(len(val)):
#             data[key][i][0] = markers[data[key][i][0]]
#
#     # Indicates the column positions for extracting from data = { ... }
#     mark, ddec3, onetep, chi = 0, 1, 2, 3
#
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#
#     ax1.set_autoscaley_on(False)
#     ax1.set_xlim([-1, 1])
#     ax1.set_ylim([-1, 1])
#     ax1.set(adjustable='box-forced', aspect='equal')
#
#     # Add black diagonal line for reference.
#     line = mlines.Line2D([0, 1], [0, 1], color='k')
#     transform = ax1.transAxes
#     line.set_transform(transform)
#     ax1.add_line(line)
#
#     [ax1.scatter([float(col[onetep]) for col in val], [float(col[chi]) for col in val],
#                  marker='x') for key, val in data.items()]
#
#     ax1.set_xlabel('ONETEP')
#     ax1.set_ylabel('QUBEKit (IPCM, DDEC3, 14/3 CHI)')
#     ax1.grid(True)
#     ax1.annotate('R² = 0.9803', xy=(-0.8, 0.8))
#
#     # Best fit
#     x = arange(-1, 2)
#     y = 0.9331 * x - 0.002
#     b, m = polyfit(x, y, 1)
#     ax1.plot(x, b + m * x, '-')
#
#     ax2.set_autoscaley_on(False)
#     ax2.set_xlim([-1, 1])
#     ax2.set_ylim([-1, 1])
#     ax2.set(adjustable='box-forced', aspect='equal')
#
#     line = mlines.Line2D([0, 1], [0, 1], color='k')
#     transform = ax2.transAxes
#     line.set_transform(transform)
#     ax2.add_line(line)
#
#     [ax2.scatter([float(col[ddec3]) for col in val], [float(col[chi]) for col in val],
#                  marker='x') for key, val in data.items()]
#
#     ax2.set_xlabel('QUBEKit (IPCM, DDEC3)')
#     ax2.set_ylabel('QUBEKit (IPCM, DDEC3, 14/3 CHI)')
#     ax2.grid(True)
#     ax2.annotate('R² = 0.9987', xy=(-0.8, 0.8))
#
#     # Best fit
#     x = arange(-1, 2)
#     y = 1.0001 * x - 2 * (10 ** -8)
#     b, m = polyfit(x, y, 1)
#     ax2.plot(x, b + m * x, '-')
#
#     plt.legend(['Equally Charged', 'LSR'] + [str(key) for key, val in data.items()], loc='lower right')
#
#     plt.show()

defaults_dict = {'charge': 0,
                 'multiplicity': 1,
                 'config': 'default_config'}

qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
config_dict = [defaults_dict, qm, fitting, descriptions]


def main():

    aa = Protein('aceleunme.pdb')

    # Rather than using the parametrisation classes:
    aa.atom_types()
    aa.residues()
    aa.construct_xml_dicts()

    XML(aa, input_file='qube.xml')

    aa.write_parameters()

    # aa.harmonic_bond_force()
    # aa.harmonic_angle_force()
    # aa.periodic_torsion_force()
    # aa.improper_torsion_force()

    # Need a ONETEP file to gather LJ params
    # lj = LJ(aa, config_dict)
    # aa.NonbondedForce = lj.calculate_non_bonded_force()


if __name__ == '__main__':

    main()
