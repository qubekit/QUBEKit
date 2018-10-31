#! /usr/bin/env python

from engines import PSI4
from ligand import Ligand
from QUBEKit.modseminario import modified_seminario_method

# def gather_charges():
#     """Takes the TheoryTests files and extracts the net charge as a tuple with the molecule + functional
#     For example, opens the benzene_PBE0_001 folder, finds the charges file from DDEC6,
#     finds the net charge from the carbon atoms and finally returns them as:
#     {net charge, benzene_PBE0}
#
#     These charges can then be output to a graph."""
#
#     from operator import itemgetter
#
#     molecules = ['/benzene', '/methane', '/ethane', '/acetone', '/methanol']
#     charges_list = []
#
#     for root, dirs, files in os.walk('./TheoryTests'):
#         for file in files:
#             for i in range(len(molecules)):
#                 if molecules[i] in root:
#                     if file.startswith('DDEC6_even_tempered_net'):
#                         name = file
#                         # print(root + '/' + name)
#                         with open(root + '/' + name, 'r') as charge_file:
#                             net_charge = 0
#                             lines = charge_file.readlines()
#                             for count, line in enumerate(lines):
#                                 if line[0:2] == 'C ':
#                                     net_charge += float(line.split()[4])
#                             # Find average charge
#                             #         net_charge /= (count + 1)
#                         charges_list.append([molecules[i][1:], root.split('_')[-2], round(net_charge, 4)])
#                         # Sort list by molecule
#                         charges_list = sorted(charges_list, key=itemgetter(0))
#                         # Sort list by functional
#                         charges_list = sorted(charges_list, key=itemgetter(1))
#
#     return np.array(charges_list)
#
#
# def plot_charges(charges=gather_charges()):
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     N = 5
#     ind = np.arange(N)  # The x locations for the groups
#     width = 0.18  # The width of the bars
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#
#     # Segregate data according to molecule and functional
#     charges = [float(charge[2]) for charge in charges]
#     B3LYP = charges[0:5]
#     BB1K = charges[5:10]
#     PBE = charges[10:15]
#     wB97X_D = charges[15:20]
#
#     # Set group separation
#     rects1 = ax.bar(ind, B3LYP, width)
#     rects2 = ax.bar(ind + width, BB1K, width)
#     rects3 = ax.bar(ind + width * 2, PBE, width)
#     rects4 = ax.bar(ind + width * 3, wB97X_D, width)
#
#     plt.title('Net Charges Across Carbon Atoms')
#     ax.set_ylabel('Net Charges')
#     ax.set_xticks(ind + width)
#     ax.set_xticklabels(('Acetone', 'benzene', 'ethane', 'methane', 'methanol'))
#     ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('B3LYP', 'BB1K', 'PBE', 'wB97X-D'))
#
#     ax.yaxis.grid(which="both", linewidth=0.7)
#
#     plt.show()


# print(gather_charges())
# print(plot_charges())

defaults_dict = {'charge': 0, 'multiplicity': 1,
                 'bonds engine': 'psi4', 'charges engine': 'chargemol',
                 'ddec version': 6, 'geometric': True, 'solvent': False,
                 'run number': '999', 'config': 'default_config'}


file = 'ethane.pdb'
mol = Ligand(file)
# print(mol)
if defaults_dict['bonds engine'] == 'psi4':
    QMengine = PSI4(mol, defaults_dict['config'], defaults_dict['geometric'], defaults_dict['solvent'])

    #os.chdir('methane_999')

    if defaults_dict['geometric']:
        #print('writing the input files')
        #QMengine.geo_gradiant(0, 1)
        print('extracting the optimized structure')
        mol = mol.read_xyz_geo()
        print(mol)
        #print('now write the hessian input file and run')
        #QMengine.generate_input(0, 1, QM=True)
        # print(mol.get_bond_lengths(QM=True))
        print('extracting hessian')
        mol = QMengine.hessian()

        def isSymmetric(mat, N):
            for i in range(N):
                for j in range(N):
                    if (mat[i,j] != mat[j,i]):
                        return False
            return True
        print(isSymmetric(mol.hessian, 15))
        print(len(mol.hessian))
        # need to give the vib scalling from the configs folder
        modified_seminario_method(0.957, mol)



# QMengine.generate_input(defaults_dict['charge'], defaults_dict['multiplicity'])
