#! /usr/bin/env python

# from QUBEKit.engines import PSI4, Gaussian
# from QUBEKit.ligand import Ligand
# from QUBEKit.parametrisation import OpenFF, XML, AnteChamber
# from QUBEKit.smiles import smiles_mm_optimise, smiles_to_pdb
# from QUBEKit.helpers import config_loader
# from QUBEKit.dihedrals import TorsionScan
#

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
                 'ddec version': 6, 'geometric': True, 'solvent': None,
                 'run number': '999', 'config': 'default_config'}

# file = 'ethane.pdb'
# mol = Ligand(file)
#
# if defaults_dict['bonds engine'] == 'psi4':
#     QMengine = PSI4(mol, defaults_dict['config'], defaults_dict['geometric'], defaults_dict['solvent'])
#
#     os.chdir('QUBEKit_2018_11_01_ethane_999')


# def mode_check(g09_modes, mol):
#     """"Compare the modes calculated by g09 and psi4 calculate the % error and MUE taking g09 as ref
#     no scalling as they are calculated at the same theroy"""
#     import numpy as np
#
#     MMfreq = np.array(mol.modes)
#     N = len(list(mol.topology.nodes))
#     print(N)
#     front = 1 / (3 * N - 6)
#     QMfreq = np.array(g09_modes)
#     bot =  QMfreq
#     # print(bot)
#     top = bot - MMfreq
#     # print(top)
#     mean_percent = 100 * front * sum(abs(top / bot))
#     mean_error = front * sum(abs(top))
#     print("Mean percentage error across frequencies = %5.3f" % (mean_percent))
#     print("Mean unsigned error across frequencies = %6.3f" % (mean_error))
#
#
# file = 'ethane.pdb'
# mol = Ligand(file)
# # print(mol)
# MMengine = 'OpenMM' #  place holder for now
# if defaults_dict['bonds engine'] == 'psi4':
#     QMengine = PSI4(mol, defaults_dict)

    # os.chdir('QUBEKit_2018_10_31_methane_666')

    # if defaults_dict['geometric']:
        # print('writing the input files running geometric')
        # QMengine.geo_gradiant(0, 1)

        # QMengine.generate_input(0, 1)
        # print('extracting the optimized structure')
        # mol.read_xyz_geo()
        # mol = QMengine.optimised_structure()
        # print(mol)
        # print('now write the hessian input file and run')
        # QMengine.generate_input(0, 1, QM=True, hessian=True)
        # print(mol.get_bond_lengths(QM=True))
        # print('extracting hessian')
        # mol = QMengine.hessian()
        # mol.hessian = input_data_processing_g09()
        # print(mol)
        # def isSymmetric(mat, N):
        #     for i in range(N):
        #         for j in range(N):
        #             if (mat[i,j] != mat[j,i]):
        #                 return False
        #     return True
        # print(isSymmetric(mol.hessian, 15))
        #print(len(mol.hessian))

        # need to give the vib scalling from the configs folder
        # modified_seminario_method(0.957, mol)
        # g09_B3LYP_modes = [307.763,  826.0159, 827.2216, 996.2495, 1216.8186, 1217.6571, 1407.3454, 1422.7768, 1503.0658, 1503.4704, 1505.2479, 1505.6004, 3024.1788, 3024.2981, 3069.7309, 3069.9317, 3094.7834, 3094.9185]
        # g09_PBE_modes = [302.8884, 798.9961, 800.5292, 990.4741, 1172.8610, 1173.9200,  1353.9015, 1371.2520, 1453.8914, 1454.4174, 1454.5196, 1454.9089, 2966.7425, 2968.0405, 3020.3395, 3020.5401, 3045.1898, 3045.3219]
        # g09_wb97xd_modes = [306.5634, 829.3540, 834.2469, 1016.6714, 1219.6302, 1221.3450, 1406.3348, 1432.7911, 1505.4329, 1506.6761, 1511.0973, 1511.3159, 3051.1161, 3053.2269, 3110.2532, 3111.6376, 3133.7972, 3135.1898]
        # mol = QMengine.all_modes()
        # print(mol.modes)
        # print('comparing modes to g09')
        # mode_check(g09_B3LYP_modes, mol)

# mol = Ligand('3HA.pdb')
# print(mol)
#
# OpenFF(mol)
#
# print('writing parameters')
# mol.write_parameters()

# now test that the new xml file can be used in a calculation
# from simtk.openmm.app import *
# from simtk.openmm import *
# from simtk.unit import *
#
#
# print('run the molecule')
#
# pdb = PDBFile('3HA.pdb')
# modeller = Modeller(pdb.topology, pdb.positions)
# forcefield = ForceField('3HA.xml')
# system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff,  constraints=None)
# integrator = LangevinIntegrator(
#         300*kelvin, 5 / picosecond,  0.0005 * picoseconds)
# simulation = Simulation(modeller.topology, system, integrator)
# simulation.context.setPositions(modeller.positions)
# #simulation = Minimize(simulation,100)
# #simulation.reporters.append(app.PDBReporter('gas_output.pdb', 1000))
# #simulation.reporters.append(app.StateDataReporter('gas.txt', 1000, step=True, temperature=True, potentialEnergy=True, density=True,totalSteps=10000, totalEnergy=True))
# #simulation.step(6000000)
# state = simulation.context.getState(getEnergy=True)
# print(state.getPotentialEnergy())
#
# print('Loading Molecule')
# ethane = Ligand('ethane.pdb')
# ethane.pickle(state='bonds')
# print('pickled bonds')
# ethane.pickle(state='charges')
# print('pickled charges')
# ethane.pickle(state='torsions')
# print('pickled torsions')
#
# from helpers import unpickle
#
# states = unpickle('.ethane_states')
# for mol in states:
#     print(states[mol])
def main():

    from QUBEKit.helpers import Configure
    from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
    from QUBEKit.ligand import Ligand
    from QUBEKit.engines import PSI4
    from os import chdir
    from QUBEKit.parametrisation import OpenFF, XML, AnteChamber
    from shutil import  copy

    defaults_dict = {'charge': 0,
                              'multiplicity': 1,
                              'config': 'default_config'}
    print('loading configs')
    qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])

    config_dict = [defaults_dict, qm, fitting, descriptions]
    print('loading molecule')
    scan = (1,2)
    mol = Ligand('ethane.pdb')
    print('paramiterising')
    # OpenFF(mol)
    # XML(mol)
    AnteChamber(mol)
    mol.write_parameters()
    copy('ethane.xml', 'test.xml')
    print('loading engine')
    QMengine = PSI4(mol, config_dict)
    print('loading scanner')
    scanner = TorsionScan(mol, QMengine)
    print('getting energys')
    scan = (1,2)
    copy('ethane.pdb', f'SCAN_{scan}')
    copy('ethane.xml', f'SCAN_{scan}')
    chdir(f'SCAN_{scan}')
    scanner.get_energy(mol.scan_order[0])
    opt = TorsionOptimiser(mol, QMengine, config_dict, opls=False)
    opt.run()
    print(mol.PeriodicTorsionForce)




if __name__ == '__main__':
    main()



