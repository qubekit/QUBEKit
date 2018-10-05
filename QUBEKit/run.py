#!/usr/bin/env python


"""
Given a list of commands, such as: "bonds", "-h", "psi4" some are taken as single word commands.
Others however, such as changing defaults: ("-c", "0") ("-m", "1") etc are taken as tuple commands.
All tuple commands are preceeding by a "-", while single word commands are not.
An error is raised for hanging commands e.g. "-c", "1" or "-h".
Single-word commands are taken in any order, assuming they are given, tuple commands are taken as ("-key", "val").
All commands are optional. If nothing at all is given, the program will run entirely with defaults.
Help can be called with the tuple command "-help" or just "-h" followed by the argument the user wants help with.
Files to be analysed must be written with their file extension (.pdb) attached or they will not be recognised commands.
All commands should be given in lower case.

Some examples:

Getting help with the bonds command:
python run.py -h bonds

Running a charges analysis with a non-default charge of 1 and the default charge engine (chargemol):
Note, ordering does not matter as long as tuples commands are together.
python run.py molecule.pdb charges -c 1
python run.py -c 1 molecule.pdb charges

Running a bonds analysis (default) with a non-default engine (g09):
python run.py molecule.pdb g09
More explicitly:
python run.py molecule.pdb bonds g09

Help is always checked for first and will stop any further action until the script is re-called.

The program will tell the user which defaults are being used, and which commands were given.
Errors will be raised for any invalid commands and the program will not run.

Each time the program runs, a new log file will be created with today's date and a run number.
This log number is also used to create the directory where all the job files and outputs are stored.
If using QUBEKit multiple times per day, it is therefore necessary to update the 'run number' else the log files and directory may be overwritten.
This can be done with the command:
python run.py -log #
where '#' is any number or string you like.
"""


from sys import argv as cmdline
import bonds
import charges
from os import system, mkdir
import subprocess


defaults_dict = {'charge': 0, 'multiplicity': 1,
                 'bonds engine': 'psi4', 'charges engine': 'chargemol',
                 'ddec version': 6, 'geometric': True, 'solvent': True,
                 'run number': '999', 'config': 'default_config'}

# TODO Complete help dict to cover everything.
help_dict = {'bonds': 'Bonds help, default={}'.format(defaults_dict['bonds engine']),
             'charges': 'Charges help, default={}'.format(defaults_dict['charges engine']),
             'dihedrals': 'Dihedrals help, default=???',
             'smiles': 'Enter the smiles code for the molecule.',
             '-c': 'The charge of the molecule, default={}'.format(defaults_dict['charge']),
             '-m': 'The multiplicity of the molecule, default={}'.format(defaults_dict['multiplicity']),
             'qubekit': 'Welcome to QUBEKit\n{}'.format(__doc__),
             'ddec': 'It is recommended to use the default DDEC6, however DDEC3 is available.',
             'log': 'The name applied to the log files and directory for the current analysis.'}


def get_opt_structure_geo(file, molecule, molecule_name):
    """Obtains the optimised structure from geometric."""

    qm, fitting, paths = bonds.config_loader(defaults_dict['config'])

    if defaults_dict['geometric']:
        print('Writing psi4 style input file for geometric')
        bonds.pdb_to_psi4_geo(file, molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'])

    else:
        bonds.pdb_to_psi4(file, molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'], qm['threads'])

    # Now run the optimisation in psi4 using geometric
    run = input('would you like to run the optimisation?\n>')
    if any(s in run.lower() for s in ('y', 'yes')):

        if defaults_dict['geometric']:

            print('Calling geometric and psi4 to optimise.')

            with open('log.txt', 'w+') as log:
                subprocess.call('geometric-optimize --psi4 {}.psi4in --nt {}'.format(molecule_name, qm['threads']), shell=True, stdout=log)

            # Make sure optimisation has finished
            opt_molecule = bonds.get_molecule_from_psi4()
            return opt_molecule


def bonds_psi4(file, molecule, molecule_name):
    """Run function for psi4 bonds.
    Will create all relevant files according to whether geometric is enabled or not."""

    qm, fitting, paths = bonds.config_loader(defaults_dict['config'])

    if defaults_dict['geometric']:
        print('Writing psi4 style input file for geometric')
        bonds.pdb_to_psi4_geo(file, molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'])

    else:
        bonds.pdb_to_psi4(file, molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'], qm['threads'])

    # Now run the optimisation in psi4 using geometric
    run = input('would you like to run the optimisation?\n>')
    if any(s in run.lower() for s in ('y', 'yes')):

        if defaults_dict['geometric']:

            print('Calling geometric and psi4 to optimise.')

            with open('log.txt', 'w+') as log:
                subprocess.call('geometric-optimize --psi4 {}.psi4in --nt {}'.format(molecule_name, qm['threads']), shell=True, stdout=log)

            # Make sure optimisation has finished
            opt_molecule = bonds.get_molecule_from_psi4()

            # Optimised molecule structure stored in opt_molecule; now prep psi4 file
            bonds.input_psi4(file, opt_molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'], qm['memory'])

            print('Calling psi4 to calculate frequencies and Hessian matrix.')
            subprocess.call('psi4 {}_freq.dat freq_out.dat -n {}'.format(molecule_name, qm['threads']), shell=True)

            # Move files out of temp folder and delete temp folder.
            print('Moving files to {}_{} folder.'.format(molecule_name, defaults_dict['run number']))
            system('mv {}.tmp/* .'.format(molecule_name))
            system('rmdir {}.tmp'.format(molecule_name))

            # Now check and extract the formatted hessian in N * N form
            print('Extracting Hessian.')
            form_hess = bonds.extract_hess_psi4_geo(opt_molecule)

            print('Geometric and psi4 bonds analysis successful; files ready in {}_{} directory.'.format(molecule_name, defaults_dict['run number']))

        else:
            print('Calling psi4 without geometric to optimise.')
            bonds.pdb_to_psi4(file, molecule, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'], qm['memory'])
            subprocess.call('psi4 input.dat -n {}'.format(qm['threads']))

            # Move files out of temp folder and delete temp folder.
            print('Moving files to {}_{} folder.'.format(molecule_name, defaults_dict['run number']))
            system('mv {}.tmp/* .'.format(molecule_name))
            system('rmdir {}.tmp'.format(molecule_name))

            print('Extracting Hessian.')
            form_hess = bonds.extract_hess_psi4(molecule)

            print('Extracting final optimised structure.')
            opt_molecule = bonds.extract_final_opt_struct_psi4(molecule)

            print('Psi4 bonds analysis successful; files ready in {}_{} directory.'.format(molecule_name, defaults_dict['run number']))

        freq = bonds.extract_all_modes_psi4(molecule)

        system('mv *.dat *.xyz *.txt *.psi4in {}_{}'.format(molecule_name, defaults_dict['run number']))

        return form_hess, freq, opt_molecule


def bonds_g09():

    print('g09 not currently supported.')
    pass


def charges_chargemol(file, molecule, molecule_name):
    """"Run function for chargemol charges."""

    qm, fitting, paths = bonds.config_loader(defaults_dict['config'])

    print('Using geometric regardless of defaults.')
    f_opt = get_opt_structure_geo(file, molecule, molecule_name)

    bonds.input_psi4(file, f_opt, defaults_dict['charge'], defaults_dict['multiplicity'], qm['basis'], qm['theory'], qm['threads'], 'c')

    subprocess.call('psi4 {}_charge.dat -n {}'.format(molecule_name, qm['threads']), shell=True)

    print('Running chargemol with DDEC version {}.'.format(defaults_dict['ddec version']))
    charges.charge_gen(defaults_dict['ddec version'], '/home/b8009890/Programs/chargemol_09_26_2017')

    subprocess.call('mv Dt.cube total_density.cube', shell=True)
    subprocess.call(paths['chargemol'] + '/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_serial job_control.txt', shell=True)

    print('Moving files to {} folder.'.format(molecule_name))
    system('mv {}.tmp/* .'.format(molecule_name))
    system('rmdir {}.tmp'.format(molecule_name))

    ###

    system('mv *.dat *.cube *.xyz *.txt *.out *.output *.psi4in {}_{}'.format(molecule_name, defaults_dict['run number']))

    print('Chargemol charges analysis successful; files ready in {}_{} directory.'.format(molecule_name, defaults_dict['run number']))


def charges_onetep():

    print('onetep not currently supported.')
    pass


def main():
    """Interprets commands from the terminal. Stores defaults, returns help or executes relevant functions."""

    commands = cmdline[1:]
    print('These are the commands you gave:', commands)

    for count, cmd in enumerate(commands):

        if any(s in cmd for s in ('-h', '-help')):

            if commands[count + 1] in help_dict:
                print(help_dict[commands[count + 1]])
            else:
                print('Unrecognised help command; the valid help commands are:', help_dict.keys())
            return

        if any(s in cmd for s in ('-sm', '-smiles')):

            import smiles

            smiles.smiles_to_pdb(commands[count + 1])
            return

        # Changing defaults for each test run
        if cmd == '-c':
            defaults_dict['charge'] = int(commands[count + 1])

        if cmd == '-m':
            defaults_dict['multiplicity'] = int(commands[count + 1])

        if cmd == '-ddec':
            defaults_dict['ddec version'] = int(commands[count + 1])

        if any(s in cmd for s in ('geo', 'geometric')):
            defaults_dict['geometric'] = bool(commands[count + 1])

        if cmd == 'psi4':
            defaults_dict['bonds engine'] = 'psi4'

        if cmd == 'g09':
            defaults_dict['bonds engine'] = 'g09'

        if any(s in cmd for s in ('cmol', 'chargemol')):
            defaults_dict['charges engine'] = 'chargemol'

        if cmd == 'onetep':
            defaults_dict['charges engine'] = 'onetep'

        if cmd == '-log':
            defaults_dict['run number'] = str(commands[count + 1])

        if cmd == '-config':
            defaults_dict['config'] = str(commands[count + 1])

    print('These are the current defaults:', defaults_dict, 'Please note, some values may not be used.')

    for count, cmd in enumerate(commands):

        if 'pdb' in cmd:

            file = cmd
            molecule_name = file.split('.')[count]
            print('Analysing:', molecule_name)
            molecule = bonds.read_pdb(file)

            from datetime import datetime

            log_file_name = 'QUBEKit_log_file_{date}_{name}_{num}'.format(date=datetime.now().strftime('%Y_%m_%d'), name=molecule_name, num=defaults_dict['run number'])
            with open(log_file_name + '.txt', 'a+') as log_file:
                log_file.write('Began {name} log {num} at {date}\n\n\n'.format(name=molecule_name, num=defaults_dict['run number'], date=datetime.now()))

            # Create a working directory for the molecule being analysed.
            mkdir('{}_{}'.format(molecule_name, defaults_dict['run number']))

            if 'charges' in commands:

                if defaults_dict['charges engine'] == 'chargemol':
                    print('Running chargemol charge analysis.')
                    charges_chargemol(file, molecule, molecule_name)
                    break

                elif defaults_dict['charges engine'] == 'onetep':
                    # TODO Add ONETEP engine function
                    print('Running onetep charge analysis.')
                    charges_onetep()
                    break

                else:
                    raise Exception('Invalid charge engine given. Please use either chargemol or onetep.')

            elif 'dihedrals' in commands:

                # TODO Add dihedral calcs
                print('Running dihedrals ...')

            # else run default bonds analysis
            else:

                if defaults_dict['bonds engine'] == 'psi4':
                    print('Running psi4 bonds analysis.')
                    bonds_psi4(file, molecule, molecule_name)
                    break

                elif defaults_dict['bonds engine'] == 'g09':
                    # TODO Add g09 engine function
                    print('Running g09 bonds analysis.')
                    bonds_g09()
                    break

                else:
                    raise Exception('Invalid bond engine given. Please use either psi4 or g09.')
        else:
            raise Exception('Missing valid file type or smiles command.\nPlease use zmat or pdb files and be sure to give the extension when typing the file name into the terminal.\nAlternatively, use the smiles command to generate a molecule.')


if __name__ == '__main__':

    main()