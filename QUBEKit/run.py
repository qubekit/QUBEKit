#!/usr/bin/env python

# TODO Get rid of linux-specific commands

from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.engines import PSI4, Chargemol, Gaussian, ONETEP, QCEngine, RDKit
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
from QUBEKit.parametrisation import OpenFF, AnteChamber, XML
from QUBEKit.decorators import exception_logger
from QUBEKit.helpers import mol_data_from_csv, generate_bulk_csv, append_to_log, pretty_progress, pretty_print, \
    Configure, unpickle

from collections import OrderedDict
from datetime import datetime
from functools import partial
import os
from shutil import copy, move
from subprocess import run as sub_run
import sys

import argparse

# To avoid calling flush=True in every print statement.
printf = partial(print, flush=True)


class Main:
    """
    Interprets commands from the terminal.
    Stores defaults or executes relevant functions.
    Will also create log and working directory where needed.
    See README.md for detailed discussion of QUBEKit commands or method docstrings for specifics of their functionality.
    """

    def __init__(self):

        self.file = None
        self.log_file = 'QUBEKit_log.txt'

        self.start_up_msg = ('If QUBEKit ever breaks or you would like to view timings and loads of other info, '
                             'view the log file.\nOur documentation (README.md) '
                             'also contains help on handling the various commands for QUBEKit.\n')

        # Call order of the analysing methods.
        # Slices of this dict are taken when changing the start and end points of analyses.
        self.order = OrderedDict([('parametrise', self.parametrise),
                                  ('mm_optimise', self.mm_optimise),
                                  ('qm_optimise', self.qm_optimise),
                                  ('hessian', self.hessian),
                                  ('mod_sem', self.mod_sem),
                                  ('density', self.density),
                                  ('charges', self.charges),
                                  ('lennard_jones', self.lennard_jones),
                                  ('torsion_scan', self.torsion_scan),
                                  ('torsion_optimise', self.torsion_optimise),
                                  ('finalise', self.finalise)])

        self.immutable_order = list(self.order)

        # TODO Make QCEngine optional? i.e. don't run it automatically when psi4 is selected.
        self.engine_dict = {'psi4': PSI4, 'g09': Gaussian, 'onetep': ONETEP}

        # Argparse will only return if we are doing a QUBEKit run bulk or normal
        self.args = self.parse_commands()

        self.constraints_file = ''

        # Look through the command line options and apply bulk and restart settings
        self.check_options()

        # Configs:
        self.defaults_dict = {'charge': self.args.charge,
                              'multiplicity': self.args.multiplicity,
                              'config': self.args.config_file}

        self.configs = {'qm': {}, 'fitting': {}, 'descriptions': {}}

        # Find which config is being used and store arguments accordingly
        if self.args.config_file == 'default_config':
            if not Configure.check_master():
                # Press any key to continue
                input('You must set up a master config to use QUBEKit and change the chargemol path; '
                      'press enter to edit master config. \n'
                      'You are free to change it later, with whichever editor you prefer.')

                Configure.ini_writer('master_config.ini')
                Configure.ini_edit('master_config.ini')

        self.qm, self.fitting, self.descriptions = Configure.load_config(self.defaults_dict['config'])
        self.all_configs = [self.defaults_dict, self.qm, self.fitting, self.descriptions]

        # Update the configs with any command line options
        self.config_update()

        self.continue_log() if self.args.restart is not None else self.create_log()

        # Starting a single run so print the message
        printf(self.start_up_msg)

        self.execute()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def check_options(self):
        """
        Read through the command line and handle options which affect the main execute
        like restart or start a bulk run.
        """

        # Run the bulk command and exit the code on completion
        if self.args.bulk_run:
            self.bulk_execute()

        # check if we just want to do the torsion test?
        elif self.args.torsion_test:
            # get the file
            self.file = self.args.input
            self.order = OrderedDict([('parametrise', self.parametrise), ('torsion_test', self.torsion_test),
                                      ('finalise', self.finalise)])

        # If not bulk must be main single run
        if self.args.restart:
            self.file = [file for file in os.listdir(os.getcwd()) if '.pdb' in file][0]
        else:
            self.file = self.args.input

        # Check the end points for a normal run
        start_point = self.args.restart if self.args.restart is not None else 'parametrise'
        end_point = self.args.end if self.args.end is not None else 'finalise'
        skip_list = self.args.skip if self.args.skip is not None else []

        # Create list of all keys
        stages = list(self.order)

        # Cut out the keys before the start_point and after the end_point
        # Add finalise back in if it's removed (finalise should always be called).
        stages = stages[stages.index(start_point):stages.index(end_point)] + ['finalise']

        # Redefine self.order to only contain the key, val pairs from stages
        self.order = OrderedDict(pair for pair in self.order.items() if pair[0] in set(stages))

        for pair in self.order.items():
            if pair[0] in skip_list:
                self.order[pair[0]] = self.skip
            else:
                self.order[pair[0]] = pair[1]

    def config_update(self):
        """Update the config setting using the argparse options from the command line."""

        for key, value in vars(self.args).items():
            if value is not None:
                if key in self.qm:
                    self.qm[key] = value
                elif key in self.fitting:
                    self.fitting[key] = value
                elif key in self.descriptions:
                    self.descriptions[key] = value

    @staticmethod
    def parse_commands():
        """
        Parses commands from the terminal using argparse.

        The method then ensures the correct methods are called in a logical order.
            - First: exit on the -progress, -setup
            - Second: look for bulk commands
            - Third: look for -restart and -end commands
            - Fourth: look for smiles codes and input files

        This ordering ensures that the program will always:
            - Terminate immediately if necessary;
            - Store all config changes correctly before running anything;
            - Perform bulk analysis rather than single molecule analysis (if requested);
            - Restart or end in the correct way (if requested);
            - Perform single molecule analysis using a pdb or smiles string (if requested).

        After all commands have been parsed and appropriately used, either:
            - Commands are returned, along with the relevant file name
            - The program exits
        """

        # Action classes
        class SetupAction(argparse.Action):
            """The setup action class that is called when setup is found in the command line."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when setup is called."""

                choice = int(input('You can now edit config files using QUBEKit, choose an option to continue:\n'
                                   '1) Edit a config file\n'
                                   '2) Create a new master template\n'
                                   '3) Make a normal config file\n>'))

                if choice == 1:
                    inis = Configure.show_ini()
                    name = input(f'Enter the name or number of the config file to edit\n'
                                 f'{"".join(f"{inis.index(ini)}:{ini}    " for ini in inis)}\n>')
                    # make sure name is right
                    if name in inis:
                        Configure.ini_edit(name)
                    else:
                        Configure.ini_edit(inis[int(name)])

                elif choice == 2:
                    Configure.ini_writer('master_config.ini')
                    Configure.ini_edit('master_config.ini')

                elif choice == 3:
                    name = input('Enter the name of the config file to create\n>')
                    Configure.ini_writer(name)
                    Configure.ini_edit(name)

                else:
                    raise KeyError('Invalid selection; please choose from 1, 2 or 3.')

                sys.exit()

        class CSVAction(argparse.Action):
            """The csv creation class run when the csv option is used."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when csv is called."""

                generate_bulk_csv(values)
                sys.exit()

        class ProgressAction(argparse.Action):
            """Run the pretty progress function to get the progress of all running jobs."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when progress is called."""

                pretty_progress()
                sys.exit()

        class TorsionMakerAction(argparse.Action):
            """Help the user make a torsion scan file."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when Torsion maker is called."""
                # load in the ligand molecule
                mol = Ligand(values)

                # make fake engine class
                class Engine:
                    def __init__(self):
                        self.fitting = {'increment': 15}

                # Prompt the user for the scan order
                scanner = TorsionScan(mol, Engine())
                scanner.find_scan_order()

                # Write out the scan file
                with open('QUBE_torsions.txt', 'w+') as qube:
                    qube.write('# dihedral definition by atom indices starting from 1\n#  i      j      k      l\n')
                    for scan in mol.scan_order:
                        scan_di = mol.dihedrals[scan][0]
                        qube.write(f'  {scan_di[0]:2}     {scan_di[1]:2}     {scan_di[2]:2}     {scan_di[3]:2}\n')
                print('QUBE_torsions.txt made.')

                sys.exit()

        def string_to_bool(string):
            """Convert a string to a bool for argparse use when casting to bool"""
            return True if string.lower() in ['true', 't', 'yes', 'y'] else False

        # Extract the intro the readme. this is based on the title "What is QUBEKit" and the following paragraph
        # which starts with "Users who ... "
        intro = ''
        with open(f'{"" if os.path.exists("../README.md") else "../"}../README.md') as readme:
            flag = False
            for line in readme:
                if 'Users who' in line:
                    flag = False
                if flag:
                    intro += line
                if '## What is QUBEKit' in line:
                    flag = True

        parser = argparse.ArgumentParser(prog='QUBEKit', formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=intro)

        # Add all of the command line options in the arg parser
        parser.add_argument('-c', '--charge', default=0, type=int, help='Enter the charge of the molecule, default 0.')
        parser.add_argument('-m', '--multiplicity', default=1, type=int, help='Enter the multiplicity of the '
                                                                              'molecule, default 1.')
        parser.add_argument('-ddec', '--ddec_version', choices=[3, 6], type=int,
                            help='Enter the ddec version for charge partitioning, does not effect ONETEP partitioning.')
        parser.add_argument('-geo', '--geometric', choices=[True, False], type=string_to_bool,
                            help='Turn on geometric to use this during the qm optimisations, recommended.')
        parser.add_argument('-bonds', '--bonds_engine', choices=['psi4', 'g09'],
                            help='Choose the QM code to calculate the bonded terms.')
        parser.add_argument('-charges', '--charges_engine', choices=['onetep', 'chargemol'],
                            help='Choose the method to do the charge partioning.')
        parser.add_argument('-density', '--density_engine', choices=['onetep', 'g09', 'psi4'],
                            help='Enter the name of the QM code to calculate the electron density of the molecule.')
        parser.add_argument('-solvent', '--solvent', choices=[True, False], type=string_to_bool,
                            help='Enter whether or not you would like to use a solvent.')
        # maybe separate into known solvents and IPCM constants?
        parser.add_argument('-convergence', '--convergence', choices=['GAU', 'GAU_TIGHT', 'GAU_VERYTIGHT'],
                            help='Enter the convergence criteria for the optimisation.')
        parser.add_argument('-param', '--parameter_engine', choices=['xml', 'antechamber', 'openff'],
                            help='Enter the method of where we should get the initial molecule parameters from, '
                                 'if xml make sure the xml has the same name as the pdb file.')
        parser.add_argument('-mm', '--mm_opt_method', default='openmm', choices=['openmm', 'rdkit_mff', 'rdkit_uff'],
                            help='Enter the mm optimisation method for pre qm optimisation.')
        parser.add_argument('-config', '--config_file', default='default_config', choices=Configure.show_ini(),
                            help='Enter the name of the configuration file you wish to use for this run from the list '
                                 'available, defaults to master.')
        parser.add_argument('-theory', '--theory',
                            help='Enter the name of the qm theory you would like to use.')
        parser.add_argument('-basis', '--basis',
                            help='Enter the basis set you would like to use.')
        parser.add_argument('-restart', '--restart', choices=['parametrise', 'mm_optimise', 'qm_optimise', 'hessian',
                                                              'mod_sem', 'density', 'charges', 'lennard_jones',
                                                              'torsion_scan', 'torsion_optimise'],
                            help='Enter the restart point of a QUBEKit job.')
        parser.add_argument('-end', '-end', choices=['mm_optimise', 'qm_optimise', 'hessian', 'mod_sem', 'density',
                                                     'charges', 'lennard_jones', 'torsion_scan', 'torsion_optimise',
                                                     'finalise'], help='Enter the end point of the QUBEKit job.')
        parser.add_argument('-progress', '--progress', nargs='?', const=True,
                            help='Get the current progress of a QUBEKit single or bulk job.', action=ProgressAction)
        parser.add_argument('-skip', '--skip', nargs='+', choices=['mm_optimise', 'qm_optimise', 'hessian', 'mod_sem',
                                                                   'density', 'charges', 'lennard_jones',
                                                                   'torsion_scan', 'torsion_optimise', 'finalise'],
                            help='Option to skip certain stages of the execution.')
        parser.add_argument('-tor_test', '--torsion_test', default=False, choices=[True, False], type=string_to_bool,
                            help='Enter True if you would like to run a torsion test on the chosen torsions.')
        parser.add_argument('-tor_make', '--torsion_maker', action=TorsionMakerAction,
                            help='Allow QUBEKit to help you make a torsion input file for the given molecule')

        # Add mutually exclusive groups to stop wrong combinations of options,
        # e.g. setup should not be ran with another command
        groups = parser.add_mutually_exclusive_group()
        groups.add_argument('-setup', '--setup_config', nargs='?', const=True,
                            help='Setup a new configuration or edit an existing one.', action=SetupAction)
        groups.add_argument('-sm', '--smiles', help='Enter the smiles string of a molecule as a starting point.')
        groups.add_argument('-bulk', '--bulk_run',
                            help='Enter the name of the csv file to run as bulk, bulk will use smiles unless it finds '
                                 'a molecule file with the same name.')
        groups.add_argument('-csv', '--csv_filename', action=CSVAction,
                            help='Enter the name of the csv file you would like to create for bulk runs.')
        groups.add_argument('-i', '--input', help='Enter the molecule input pdb file (only pdb so far!)')

        return parser.parse_args()

    def bulk_execute(self):
        """Run a bulk QUBEKit job in serial mode."""

        # TODO look at worker queues to maximise resource usage

        printf(self.start_up_msg)

        csv_file = self.args.bulk_run
        bulk_data = mol_data_from_csv(csv_file)

        names = list(bulk_data)

        # Store a copy of self.order which will not be mutated.
        # This allows self.order to be built up after each run.
        temp = self.order

        for name in names:
            printf(f'\nAnalysing: {name}\n')

            # Get the start and end points from the csv file, otherwise use defaults.
            start_point = bulk_data[name]['start'] if bulk_data[name]['start'] else 'parametrise'
            end_point = bulk_data[name]['end'] if bulk_data[name]['end'] else 'finalise'

            torsion_options = bulk_data[name]['torsion order']
            stages = list(temp)

            # Set stages to be the keys of self.order which will be executed (finalise is always executed).
            stages = stages[stages.index(start_point):stages.index(end_point)] + ['finalise']
            self.order = OrderedDict(pair for pair in temp.items() if pair[0] in set(stages))

            # Configs
            self.defaults_dict = bulk_data[name]
            self.qm, self.fitting, self.descriptions = Configure.load_config(self.defaults_dict['config'])
            self.all_configs = [self.defaults_dict, self.qm, self.fitting, self.descriptions]

            # If starting from the beginning, create log and pdb file then execute as normal for each run
            if start_point == 'parametrise':

                if bulk_data[name]['smiles string'] is not None:
                    smiles_string = bulk_data[name]['smiles string']
                    self.file = RDKit.smiles_to_pdb(smiles_string, name)

                else:
                    self.file = f'{name}.pdb'

                self.create_log()

            # If starting from the middle somewhere, FIND (not create) the folder, and log and pdb files, then execute
            else:
                for root, dirs, files in os.walk('.', topdown=True):
                    for dir_name in dirs:
                        if dir_name.startswith(f'QUBEKit_{name}'):
                            os.chdir(dir_name)

                # These are the files in the active directory, search for the pdb.
                self.file = [file for file in os.listdir(os.getcwd()) if '.pdb' in file][0]

                self.continue_log()

            # if we have a torsion order add it here
            self.execute(torsion_options)
            os.chdir('../')

        sys.exit('\nFinished bulk run. Use the command -progress to view which stages have completed.')

    def create_log(self):
        """
        Creates the working directory for the job as well as the log file.
        This log file is then extended when:
            - decorators.timer_logger wraps a called method;
            - helpers.append_to_log() is called;
            - helpers.pretty_print() is called with to_file set to True;
            - decorators.exception_logger_decorator() wraps a function / method which throws an exception.
        """

        date = datetime.now().strftime('%Y_%m_%d')

        # Define name of working directory.
        # This is formatted as 'QUBEKit_molecule name_yyyy_mm_dd_log_string'.
        dir_name = f'QUBEKit_{self.file[:-4]}_{date}_{self.descriptions["log"]}'

        # Back up stuff:
        #   If you can't make a dir because it exists, back it up.
        #   True for working dirs, and the backup folder itself.
        try:
            os.mkdir(dir_name)

        except FileExistsError:
            try:
                os.mkdir('QUBEKit_backups')
                printf('Making backup folder: QUBEKit_backups')
            except FileExistsError:
                # Backup folder already made
                pass
            finally:
                count = 1
                while os.path.exists(f'QUBEKit_backups/{dir_name}_{str(count).zfill(3)}'):
                    count += 1

                move(dir_name, f'QUBEKit_backups/{dir_name}_{str(count).zfill(3)}')
                printf(f'Moving directory: {dir_name} to backup folder')
                os.mkdir(dir_name)

        # Finally, having made any necessary backups, move files and change to working dir.
        finally:
            # Copy active pdb into new directory.
            abspath = os.path.abspath(self.file)
            copy(abspath, f'{dir_name}/{self.file}')
            os.chdir(dir_name)

        # TODO Better handling of constraints files; what if we have more than one?
        for root, dirs, files in os.walk('.', topdown=True):
            for file in files:
                if 'constraints.txt' in file:
                    self.constraints_file = os.path.abspath(f'{root}/{file}')

        # Find external files
        copy_files = [f'{self.file[:-4]}.xml', 'QUBE_torsions.txt']
        for file in copy_files:
            try:
                copy(f'../{file}', file)
            except FileNotFoundError:
                pass

        with open(self.log_file, 'w+') as log_file:
            log_file.write(f'Beginning log file; the time is: {datetime.now()}\n\n\n')

        self.log_configs()

    def continue_log(self):
        """
        In the event of restarting an analysis, find and append to the existing log file
        rather than creating a new one.
        """

        with open(self.log_file, 'a+') as log_file:
            log_file.write(f'\n\nContinuing log file from previous execution; the time is: {datetime.now()}\n\n\n')

        self.log_configs()

    def log_configs(self):
        """
        Writes the runtime and file-based defaults to a log file.
        Adds some fluff like the molecule name and time.
        """

        with open(self.log_file, 'a+') as log_file:

            log_file.write(f'Analysing: {self.file[:-4]}\n\n')

            log_file.write('The runtime defaults are:\n\n')
            for key, val in vars(self.args).items():
                if val is not None:
                    log_file.write(f'{key}: {val}\n')
            log_file.write('\n')

            log_file.write('The config file defaults being used are:\n\n')
            for config in self.all_configs[1:]:
                for key, var in config.items():
                    log_file.write(f'{key}: {var}\n')
                log_file.write('\n')
            log_file.write('\n')

    def stage_wrapper(self, start_key, begin_log_msg='', fin_log_msg='', torsion_options=None):
        """
        Firstly, check if the stage start_key is in self.order; this tells you if the stage should be called or not.
        If it isn't in self.order:
            - Do nothing
        If it is:
            - Unpickle the ligand object at the start_key stage
            - Write to the log that something's about to be done (if specified)
            - Do the thing
            - Write to the log that something's been done (if specified)
            - Pickle the ligand object again with the next_key marker as its stage
        """

        mol = unpickle()[start_key]

        # if we have a torsion options dictionary pass it to the molecule
        if torsion_options is not None:
            mol = self.store_torsions(mol, torsion_options)

        skipping = False
        if self.order[start_key] == self.skip:
            printf(f'Skipping stage: {start_key}')
            append_to_log(f'skipping stage: {start_key}')
            skipping = True
        else:
            if begin_log_msg:
                printf(f'{begin_log_msg}...', end=' ')

        home = os.getcwd()

        folder_name = f'{self.immutable_order.index(start_key) + 1}_{start_key}'
        # Make sure you don't get an error if restarting
        try:
            os.mkdir(folder_name)
        except FileExistsError:
            pass
        finally:
            os.chdir(folder_name)

        self.order[start_key](mol)
        self.order.pop(start_key, None)
        os.chdir(home)

        # Begin looping through self.order, but return after the first iteration.
        for key in self.order:
            next_key = key
            if fin_log_msg and not skipping:
                printf(fin_log_msg)

            mol.pickle(state=next_key)
            return next_key

    def parametrise(self, molecule):
        """Perform initial molecule parametrisation using OpenFF, Antechamber or XML."""

        # First copy the pdb and any other files into the folder
        copy(f'../{molecule.filename}', f'{molecule.filename}')

        # Parametrisation options:
        param_dict = {'openff': OpenFF, 'antechamber': AnteChamber, 'xml': XML}

        # If we are using xml we have to move it
        if self.fitting['parameter_engine'] == 'xml':
            copy(f'../{molecule.name}.xml', f'{molecule.name}.xml')

        # Perform the parametrisation
        param_dict[self.fitting['parameter_engine']](molecule)

        append_to_log(f'Parametrised molecule with {self.fitting["parameter_engine"]}')

        return molecule

    def mm_optimise(self, molecule):
        """
        Use an mm force field to get the initial optimisation of a molecule

        options
        ---------
        RDKit MFF or UFF force fields can have strange effects on the geometry of molecules

        Geometric / OpenMM depends on the force field the molecule was parameterised with gaff/2, OPLS smirnoff.
        """

        # Check which method we want then do the optimisation
        # TODO if we don't want geometric we can do a quick native openmm full optimisation?
        if self.args.mm_opt_method == 'openmm':
            # Make the inputs
            molecule.write_pdb(input_type='input')
            molecule.write_parameters()
            # Run geometric
            with open('log.txt', 'w+') as log:
                sub_run(f'geometric-optimize --reset --epsilon 0.0 --maxiter 500 --qccnv --pdb {molecule.name}.pdb '
                        f'--openmm {molecule.name}.xml {self.constraints_file}', shell=True, stdout=log, stderr=log)

            # Read the xyz traj and store the frames
            molecule.read_xyz(f'{molecule.name}_optim.xyz')
            # Store the last from the traj as the mm optimised structure
            molecule.molecule['mm'] = molecule.molecule['traj'][-1]

        else:
            # Run an RDKit optimisation with the right FF
            rdkit_ff = {'rdkit_mff': 'MFF', 'rdkit_uff': 'UFF'}
            molecule.filename = RDKit.mm_optimise(molecule.filename, ff=rdkit_ff[self.args.mm_opt_method])

        append_to_log(f'mm_optimised the molecule with {self.args.mm_opt_method}')

        return molecule

    def qm_optimise(self, molecule):
        """Optimise the molecule with or without geometric."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)

        if self.qm['geometric'] and self.qm['bonds_engine'] == 'psi4':

            # Optimise the structure using QCEngine with geometric and psi4
            qceng = QCEngine(molecule, self.all_configs)
            result = qceng.call_qcengine('geometric', 'gradient', input_type='mm')
            # Check if converged and get the geometry
            if result['success']:
                # Load all of the frames into the molecules trajectory holder
                molecule.read_geometric_traj(result['trajectory'])
                # store the last frame as the qm optimised structure
                molecule.molecule['qm'] = molecule.molecule['traj'][-1]
                # Write out the trajectory file
                molecule.write_xyz(input_type='traj', name=f'{molecule.name}_opt')
                molecule.write_xyz(input_type='qm', name='opt')

            else:
                sys.exit('Molecule not optimised.')

        else:
            converged = qm_engine.generate_input(input_type='mm', optimise=True)

            # Check the exit status of the job; if failed restart the job up to 2 times
            restart_count = 1
            while not converged and restart_count < 3:
                append_to_log(f'{self.qm["bonds_engine"]} optimisation failed; restarting', msg_type='minor')
                converged = qm_engine.generate_input(input_type='mm', optimise=True, restart=True)
                restart_count += 1

            if not converged:
                sys.exit(f'{self.qm["bonds_engine"]} optimisation did not converge after 3 restarts; check log file.')

            molecule.molecule['qm'], molecule.qm_energy = qm_engine.optimised_structure()
            molecule.write_xyz(input_type='qm', name='opt')

        append_to_log(f'qm_optimised structure calculated{" with geometric" if self.qm["geometric"] else ""}')

        return molecule

    def hessian(self, molecule):
        """Using the assigned bonds engine, calculate and extract the Hessian matrix."""

        # TODO Because of QCEngine, nothing is being put into the hessian folder anymore

        molecule.get_bond_lengths(input_type='qm')

        # Check what engine we want to use
        if self.qm['bonds_engine'] == 'g09':
            qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
            qm_engine.generate_input(input_type='qm', hessian=True)
            molecule.hessian = qm_engine.hessian()

        else:
            qceng = QCEngine(molecule, self.all_configs)
            molecule.hessian = qceng.call_qcengine('psi4', 'hessian', input_type='qm')

        append_to_log(f'Hessian calculated using {self.qm["bonds_engine"]}')

        return molecule

    def mod_sem(self, molecule):
        """Modified Seminario for bonds and angles."""

        mod_sem = ModSeminario(molecule, self.all_configs)
        mod_sem.modified_seminario_method()

        append_to_log('Mod_Seminario method complete')

        return molecule

    def density(self, molecule):
        """Perform density calculation with the qm engine."""

        qm_engine = self.engine_dict[self.qm['density_engine']](molecule, self.all_configs)
        qm_engine.generate_input(input_type='qm', density=True, solvent=self.qm['solvent'])

        if self.qm['density_engine'] == 'g09':
            append_to_log('Density analysis complete')
        else:
            # If we use onetep we have to stop after this step
            append_to_log('Density analysis file made for ONETEP')

            # Now we have to edit the order to end here.
            self.order = OrderedDict([('density', self.density), ('charges', self.skip), ('lennard_jones', self.skip),
                                      ('torsion_scan', self.torsion_scan), ('pause', self.pause)])

        return molecule

    def charges(self, molecule):
        """Perform DDEC calculation with Chargemol."""

        # TODO add option to use chargemol on onetep cube files.
        copy(f'../6_density/{molecule.name}.wfx', f'{molecule.name}.wfx')
        c_mol = Chargemol(molecule, self.all_configs)
        c_mol.generate_input()

        append_to_log(f'Charge analysis completed with Chargemol and DDEC{self.qm["ddec_version"]}')

        return molecule

    def lennard_jones(self, molecule):
        """Calculate Lennard-Jones parameters, and extract virtual sites."""

        os.system('cp ../7_charges/DDEC* .')
        lj = LennardJones(molecule, self.all_configs)
        molecule.NonbondedForce = lj.calculate_non_bonded_force()

        append_to_log('Lennard-Jones parameters calculated')

        return molecule

    def torsion_scan(self, molecule):
        """Perform torsion scan."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        scan = TorsionScan(molecule, qm_engine)

        # Try to find a scan file; if none provided and more than one torsion detected: prompt user
        try:
            copy('../../QUBE_torsions.txt', 'QUBE_torsions.txt')
            scan.find_scan_order(file='QUBE_torsions.txt')
        except FileNotFoundError:
            scan.find_scan_order()

        scan.start_scan()

        append_to_log('Torsion_scans complete')

        return molecule

    def torsion_optimise(self, molecule):
        """Perform torsion optimisation."""

        # TODO get the combination rule from xml file.
        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        opt = TorsionOptimiser(molecule, qm_engine, self.all_configs, combination=molecule.combination,
                               refinement=self.fitting['refinement_method'], vn_bounds=self.fitting['tor_limit'])
        opt.run()

        append_to_log('Torsion_optimisations complete')

        return molecule

    @staticmethod
    def finalise(molecule):
        """
        Make the xml and pdb file print the ligand object to terminal (in abbreviated form) and to the log file
        after getting the rdkit descriptors.
        """

        # write the pdb file and xml file to the folder
        molecule.write_pdb()
        molecule.write_parameters()

        # get the molecule descriptors from rdkit
        molecule.descriptors = RDKit.rdkit_descriptors(molecule.filename)

        pretty_print(molecule, to_file=True)
        pretty_print(molecule)

        return molecule

    @staticmethod
    def skip(molecule):
        """A blank method that does nothing to that stage but adds the pickle points to not break the flow."""

        return molecule

    @staticmethod
    def pause():
        """
        Pause the analysis when using ONETEP so we can come back into the workflow
        without breaking the pickling process.
        """

        printf('QUBEKit stopping at onetep step!\n To continue please move the ddec.onetep file and xyz file to the '
               'density folder and use -restart lennard_jones.')

        return

    @staticmethod
    def store_torsions(molecule, torsions_list):
        """
        Take the molecule object and the list of torsions and convert them to rotatable centres. Then, put them in the
        scan order object.
        """

        if torsions_list is not None:

            scan_order = []
            for torsion in torsions_list:
                tor = tuple(atom for atom in torsion.split('-'))
                # convert the string names to the index names and get the core indexed from 1 to match the topology
                core = (molecule.atom_names.index(tor[1]) + 1, molecule.atom_names.index(tor[2]) + 1)

                if core in molecule.rotatable:
                    scan_order.append(core)
                elif reversed(core) in molecule.rotatable:
                    scan_order.append(reversed(core))

            molecule.scan_order = scan_order

        return molecule

    def torsion_test(self, molecule):
        """Take the molecule and do the torsion test method."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        opt = TorsionOptimiser(molecule, qm_engine, self.all_configs, combination=molecule.combination,
                               refinement=self.fitting['refinement_method'], vn_bounds=self.fitting['tor_limit'])

        # test the torsions!
        opt.torsion_test()

        print('Torsion testing done!')

    @exception_logger
    def execute(self, torsion_options=None):
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order.
        Exceptions are added to log (if raised).
        Will also add the extra options dictionary to the molecule.
        """

        # split the torsion list
        if torsion_options is not None:
            torsion_options = torsion_options.split(',')

        # Check if starting from the beginning; if so:
        if 'parametrise' in self.order:
            # Initialise ligand object fully before pickling it
            molecule = Ligand(self.file)
            molecule.constraints_file = self.constraints_file

            # If there are extra options add them to the molecule
            if torsion_options is not None:
                molecule = self.store_torsions(molecule, torsion_options)
            molecule.pickle(state='parametrise')

        # Perform each key stage sequentially adding short messages (if given) to terminal to show progress.
        # Longer messages should be written inside the key stages' methods using helpers.append_to_log().
        # See PSI4 class in engines for an example of where this is used.

        # The stage keys and messages
        stage_dict = {
            'parametrise': ['Parametrising molecule', 'Molecule parametrised'],
            'mm_optimise': ['Partially optimising with MM', 'Partial optimisation complete'],
            'qm_optimise': ['Optimising molecule, view .xyz file for progress', 'Molecule optimisation complete'],
            'hessian': ['Calculating Hessian matrix', 'Hessian matrix calculated and confirmed to be symmetric'],
            'mod_sem': ['Calculating bonds and angles with modified Seminario method', 'Bonds and angles calculated'],
            'density': [f'Performing density calculation with {self.qm["density_engine"]}',
                        'Density calculation complete'],
            'charges': [f'Chargemol calculating charges using DDEC{self.qm["ddec_version"]}', 'Charges calculated'],
            'lennard_jones': ['Performing Lennard-Jones calculation', 'Lennard-Jones parameters calculated'],
            'torsion_scan': ['Performing QM-constrained optimisation with Torsiondrive',
                             'Torsiondrive finished and QM results saved'],
            'torsion_optimise': ['Performing torsion optimisation', 'Torsion optimisation complete'],
            'finalise': ['Finalising analysis', 'Molecule analysis complete!'],
            'pause': ['Pausing analysis', 'Analysis paused!'],
            'skip': ['Skipping section', 'Section skipped'],
            'torsion_test': ['Testing torsion single point energies', 'Torsion testing complete']}

        # do the first stage in the order
        key = list(self.order)[0]
        next_key = self.stage_wrapper(key, stage_dict[key][0], stage_dict[key][1], torsion_options)

        # cannot use for loop as we mutate the dictionary during the loop
        while True:
            if next_key is None:
                break
            else:
                next_key = self.stage_wrapper(next_key, stage_dict[next_key][0], stage_dict[next_key][1])

            if next_key == 'pause':
                self.pause()
                break


def main():
    """This just stops the __repr__ automatically being called when the class is called."""
    Main()
