#! /usr/bin/env python

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


def main():

    from QUBEKit.helpers import Configure
    from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
    from QUBEKit.ligand import Ligand
    from QUBEKit.engines import PSI4
    from QUBEKit.mod_seminario import ModSeminario
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
    chdir('captain')
    mol = Ligand('captain.pdb')
    print('paramiterising')
    # OpenFF(mol)
    XML(mol, input_file='boss.xml')
    # AnteChamber(mol)
    # mol.write_parameters()
    # copy('complex.xml', 'test.xml')
    print('loading engine')
    QMengine = PSI4(mol, config_dict)
    print('loading scanner')
    scanner = TorsionScan(mol, QMengine)
    print('scanning')
    # scanner.start_scan()
    print('getting energies')
    chdir('SCAN_17_19/QM_torsiondrive')
    scanner.get_energy(mol.scan_order[0])
    chdir('../../')
    opt = TorsionOptimiser(mol, QMengine, config_dict, opt_method='BFGS', opls=True, refinement_method='SP', vn_bounds=20)
    opt.opt_test()
    print(mol.PeriodicTorsionForce)

def main_2():
    from QUBEKit.helpers import Configure
    from QUBEKit.ligand import Ligand
    from QUBEKit.parametrisation import XML, AnteChamber
    from os import chdir

    defaults_dict = {'charge': 0,
                     'multiplicity': 1,
                     'config': 'default_config'}
    print('loading configs')
    qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])

    config_dict = [defaults_dict, qm, fitting, descriptions]
    print('loading molecule')
    chdir('dynamics_test')
    mol = Ligand('G1_4f.pdb')
    #mol.read_pdb()

    # AnteChamber(mol)
    # mol.update()
    # mol.write_pdb('G1_4a_qube')
    XML(mol, mol2_file='G1_4f.mol2')
    # mol.write_pdb(name='test_qube')
    mol.write_parameters(name='qube_4F')

    print('done')


def main_sites():
    from QUBEKit.helpers import Configure
    from QUBEKit.ligand import Ligand
    from QUBEKit.lennard_jones import  LennardJones
    from QUBEKit.parametrisation import XML, AnteChamber
    from os import chdir

    defaults_dict = {'charge': 0,
                     'multiplicity': 1,
                     'config': 'default_config'}
    print('loading configs')
    qm, fitting, descriptions = Configure.load_config(defaults_dict['config'])
    qm['charges_engine'] = 'onetep'

    config_dict = [defaults_dict, qm, fitting, descriptions]
    print('loading molecule')
    chdir('sites_test_2')
    mol = Ligand('MOL.pdb')
    print(mol.symm_hs)
    print(mol.rotatable)
    # mol.read_xyz(name='ddec', input_type='qm')
    # mol.read_pdb()

    # AnteChamber(mol)
    # mol.update()
    # mol.write_pdb('G1_4a_qube')
    # XML(mol)
    # # mol.write_pdb(name='test_qube')
    # lj = LennardJones(mol, config_dict)
    # mol.NonbondedForce = lj.calculate_non_bonded_force()
    # mol.write_parameters(name='v_site_test')


    print('done')

if __name__ == '__main__':
    main_sites()



