#!/usr/bin/env python3

# TODO
#  Squash unnecessary arguments into self.molecule. Args such as torsion_options.
#  Better handling of torsion_options

from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
from QUBEKit.engines import PSI4, Chargemol, Gaussian, ONETEP, QCEngine, RDKit
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.ligand import Ligand
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.parametrisation import OpenFF, AnteChamber, XML
from QUBEKit.parametrisation.base_parametrisation import Parametrisation
from QUBEKit.utils import constants
from QUBEKit.utils.decorators import exception_logger
from QUBEKit.utils.exceptions import OptimisationFailed, HessianCalculationFailed
from QUBEKit.utils.helpers import mol_data_from_csv, generate_bulk_csv, append_to_log, pretty_progress, pretty_print, \
    Configure, unpickle

from collections import OrderedDict
from datetime import datetime
from functools import partial
import os
from pathlib import Path
from shutil import copy, move
import subprocess as sp
import sys

import argparse
import numpy as np


# To avoid calling flush=True in every print statement.
printf = partial(print, flush=True)


class ArgsAndConfigs:
    """
    This class will be called once for any individual or bulk run.
    As such, this class needs to:
        * Parse all commands from terminal
        * If individual:
            * Initialise molecule
            * Using proper config file (from terminal command or default)
            * Store all configs from file into Molecule object
            * Store any config changes from terminal into Molecule object
            * Call Execute (which handles: working dir, logging, order, etc; see docstring for info)

        * If bulk:
            * For each molecule:
                * Initialise molecule
                * Using proper config file (from terminal command or default)
                * Store all configs from file
                * Store any config changes from terminal
                * Call Execute
    """

    def __init__(self):
        # First make sure the config folder has been made missing for conda and pip
        home = str(Path.home())
        config_folder = f'{home}/QUBEKit_configs/'
        if not os.path.exists(config_folder):
            os.makedirs(config_folder)
            printf(f'Making config folder at: {home}')

        self.args = self.parse_commands()

        # If it's a bulk run, handle it separately
        # TODO Add .sdf as possible bulk_run, not just .csv
        if self.args.bulk_run:
            self.handle_bulk()

        # If we are doing the torsion test add the attribute to the molecule so we can catch it in execute
        if self.args.torsion_test:
            self.args.restart = 'finalise'

        if self.args.restart is not None:
            # Find the pickled checkpoint file and load it as the molecule
            try:
                self.molecule = unpickle()[self.args.restart]
            except FileNotFoundError:
                raise FileNotFoundError('No checkpoint file found!')
        else:
            if self.args.smiles:
                self.molecule = Ligand(*self.args.smiles)
            else:
                # Initialise molecule
                self.molecule = Ligand(self.args.input)

        # Find which config file is being used
        self.molecule.config = self.args.config_file

        # Handle configs which are in a file
        file_configs = Configure().load_config(self.molecule.config)
        for name, val in file_configs.items():
            setattr(self.molecule, name, val)

        # Although these may be None always, they need to be explicitly set anyway.
        setattr(self.molecule, 'restart', None)
        setattr(self.molecule, 'end', None)
        setattr(self.molecule, 'skip', None)

        # Handle configs which are changed by terminal commands
        for name, val in vars(self.args).items():
            if val is not None:
                setattr(self.molecule, name, val)

        # Now we need to remove torsion_test as it is passed from the command line
        if self.args.torsion_test is False:
            del self.molecule.torsion_test

        # Now check if we have been supplied a dihedral file and a constraints file
        if self.args.dihedral_file:
            self.molecule.read_scan_order(self.args.dihedral_file)
        if self.args.constraints_file:
            self.molecule.constraints_file = Path(self.args.constaints_file)

        # If restarting put the molecule back into the checkpoint file with the new configs
        if self.args.restart is not None:
            self.molecule.pickle(state=self.args.restart)

        # Now that all configs are stored correctly: execute.
        Execute(self.molecule)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    @staticmethod
    def parse_commands():
        """
        Parses commands from the terminal using argparse.

        Contains classes for handling actions as well as simple arg parsers for config changes.
        """

        # Action classes
        class SetupAction(argparse.Action):
            """The setup action class that is called when setup is found in the command line."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when setup is called."""

                choice = int(input('You can now edit config files using QUBEKit, choose an option to continue:\n'
                                   '1) Edit a config file\n'
                                   '2) Create a new master template\n'
                                   '3) Make a normal config file\n'
                                   '4) Cancel\n>'))

                if choice == 1:
                    inis = Configure().show_ini()
                    name = input(f'Enter the name or number of the config file to edit\n'
                                 f'{"".join(f"{inis.index(ini)}:{ini}    " for ini in inis)}\n>')
                    # make sure name is right
                    if name in inis:
                        Configure().ini_edit(name)
                    else:
                        Configure().ini_edit(inis[int(name)])

                elif choice == 2:
                    Configure().ini_writer('master_config.ini')
                    Configure().ini_edit('master_config.ini')

                elif choice == 3:
                    name = input('Enter the name of the config file to create\n>')
                    Configure().ini_writer(name)
                    Configure().ini_edit(name)

                else:
                    sys.exit('Cancelling setup; no changes made. '
                             'If you accidentally entered the wrong key, restart with QUBEKit -setup')

                sys.exit()

        class CSVAction(argparse.Action):
            """The csv creation class run when the csv option is used."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when csv is called."""

                generate_bulk_csv(*values)
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

                # Prompt the user for the scan order
                scanner = TorsionScan(mol)
                scanner.find_scan_order()

                # Write out the scan file
                with open(f'{mol.name}.dihedrals', 'w+') as qube:
                    qube.write('# dihedral definition by atom indices starting from 0\n#  i      j      k      l\n')
                    for scan in mol.scan_order:
                        scan_di = mol.dihedrals[scan][0]
                        qube.write(f'  {scan_di[0]:2}     {scan_di[1]:2}     {scan_di[2]:2}     {scan_di[3]:2}\n')
                printf(f'{mol.name}.dihedrals made.')

                sys.exit()

        def string_to_bool(string):
            """Convert a string to a bool for argparse use when casting to bool"""
            return True if string.lower() in ['true', 't', 'yes', 'y'] else False

        intro = 'Welcome to QUBEKit! For a list of possible commands, use the help command: -h.' \
                'Alternatively, take a look through our github page for commands, recipes and common problems:' \
                'http://github.com/qubekit/QUBEKit'
        parser = argparse.ArgumentParser(prog='QUBEKit', formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description=intro)

        # Add all of the command line options in the arg parser
        parser.add_argument('-c', '--charge', type=int, help='Enter the charge of the molecule, default 0.')
        parser.add_argument('-m', '--multiplicity', type=int,
                            help='Enter the multiplicity of the molecule, default 1.')
        parser.add_argument('-threads', '--threads', type=int,
                            help='Number of threads used in various stages of analysis, especially for engines like '
                                 'PSI4, Gaussian09, etc. Value is given as an int.')
        parser.add_argument('-memory', '--memory', type=int,
                            help='Amount of memory used in various stages of analysis, especially for engines like '
                                 'PSI4, Gaussian09, etc. Value is given as an int, e.g. 6GB is simply 6.')
        parser.add_argument('-ddec', '--ddec_version', choices=[3, 6], type=int,
                            help='Enter the ddec version for charge partitioning, does not effect ONETEP partitioning.')
        parser.add_argument('-geo', '--geometric', choices=[True, False], type=string_to_bool,
                            help='Turn on geometric to use this during the qm optimisations, recommended.')
        parser.add_argument('-bonds', '--bonds_engine', choices=['psi4', 'g09', 'g16'],
                            help='Choose the QM code to calculate the bonded terms.')
        parser.add_argument('-charges', '--charges_engine', choices=['onetep', 'chargemol'],
                            help='Choose the method to do the charge partitioning.')
        parser.add_argument('-density', '--density_engine', choices=['onetep', 'g09', 'g16', 'psi4'],
                            help='Enter the name of the QM code to calculate the electron density of the molecule.')
        parser.add_argument('-solvent', '--solvent', choices=[True, False], type=string_to_bool,
                            help='Enter whether or not you would like to use a solvent.')
        # Maybe separate into known solvents and IPCM constants?
        parser.add_argument('-convergence', '--convergence', choices=['GAU', 'GAU_TIGHT', 'GAU_VERYTIGHT'],
                            help='Enter the convergence criteria for the optimisation.')
        parser.add_argument('-param', '--parameter_engine', choices=['xml', 'antechamber', 'openff', 'none'],
                            help='Enter the method of where we should get the initial molecule parameters from, '
                                 'if xml make sure the xml has the same name as the pdb file.')
        parser.add_argument('-mm', '--mm_opt_method', choices=['openmm', 'rdkit_mff', 'rdkit_uff', 'none'],
                            help='Enter the mm optimisation method for pre qm optimisation.')
        parser.add_argument('-config', '--config_file', default='default_config', choices=Configure().show_ini(),
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
                                                     'finalise'],
                            help='Enter the end point of the QUBEKit job.')
        parser.add_argument('-progress', '--progress', nargs='?', const=True,
                            help='Get the current progress of a QUBEKit single or bulk job.', action=ProgressAction)
        parser.add_argument('-skip', '--skip', nargs='+', choices=['mm_optimise', 'qm_optimise', 'hessian', 'mod_sem',
                                                                   'density', 'charges', 'lennard_jones',
                                                                   'torsion_scan', 'torsion_optimise', 'finalise'],
                            help='Option to skip certain stages of the execution.')
        parser.add_argument('-tor_test', '--torsion_test', action='store_true',
                            help='Enter True if you would like to run a torsion test on the chosen torsions.')
        parser.add_argument('-tor_make', '--torsion_maker', action=TorsionMakerAction,
                            help='Allow QUBEKit to help you make a torsion input file for the given molecule')
        parser.add_argument('-log', '--log', type=str,
                            help='Enter a name to tag working directories with. Can be any alphanumeric string.'
                                 'This helps differentiate (by more than just date) different analyses of the '
                                 'same molecule.')
        parser.add_argument('-vib', '--vib_scaling', type=float,
                            help='Enter the vibrational scaling to be used with the basis set.')
        parser.add_argument('-iters', '--iterations', type=int,
                            help='Max number of iterations for QM scan.')
        parser.add_argument('-constraints', '--constraints_file', type=str,
                            help='The name of the geometric constraints file.')
        parser.add_argument('-dihedral_file', '--dihedral_file', type=str,
                            help='The name of the qubekit/tdrive torsion file.')

        # Add mutually exclusive groups to stop certain combinations of options,
        # e.g. setup should not be run with csv command
        groups = parser.add_mutually_exclusive_group()
        groups.add_argument('-setup', '--setup_config', nargs='?', const=True,
                            help='Setup a new configuration or edit an existing one.', action=SetupAction)
        groups.add_argument('-sm', '--smiles', nargs='+',
                            help='Enter the smiles string of a molecule as a starting point.')
        groups.add_argument('-bulk', '--bulk_run',
                            help='Enter the name of the csv file to run as bulk, bulk will use smiles unless it finds '
                                 'a molecule file with the same name.')
        groups.add_argument('-csv', '--csv_filename', action=CSVAction, nargs='*',
                            help='Enter the name of the csv file you would like to create for bulk runs.'
                                 'Optionally, you may also add the maximum number of molecules per file.')
        groups.add_argument('-i', '--input', help='Enter the molecule input pdb file (only pdb so far!)')
        groups.add_argument('-version', '--version', action='version', version='2.4.0')

        return parser.parse_args()

    def handle_bulk(self):
        """
        Getting and setting configs for bulk runs is a little different, requiring this method.
        The configs are taken from the .csv, then the .ini, then the terminal.
        This is repeated for each molecule in the bulk run, then Execute is called.

        Configs cannot be changed between molecule analyses as config data is
        only loaded once at the start; -restart is required for that.
        """

        csv_file = self.args.bulk_run
        # mol_data_from_csv handles defaults if no argument is given
        bulk_data = mol_data_from_csv(csv_file)

        names = list(bulk_data)

        home = os.getcwd()

        for name in names:
            printf(f'Analysing: {name}\n')

            # Get pdb from smiles or name if no smiles is given
            if bulk_data[name]['smiles'] is not None:
                smiles_string = bulk_data[name]['smiles']
                self.molecule = Ligand(smiles_string, name)

            else:
                # TODO Different file types
                # Initialise molecule, ready to add configs to it
                self.molecule = Ligand(f'{name}.pdb')

            # Read each row in bulk data and set it to the molecule object
            for key, val in bulk_data[name].items():
                setattr(self.molecule, key, val)

            setattr(self.molecule, 'skip', [])

            # Using the config file from the .csv, gather the .ini file configs
            file_configs = Configure.load_config(self.molecule.config)
            for key, val in file_configs.items():
                setattr(self.molecule, key, val)

            # Handle configs which are changed by terminal commands
            for key, val in vars(self.args).items():
                if val is not None:
                    setattr(self.molecule, key, val)

            # Now that all configs are stored correctly: execute.
            Execute(self.molecule)

            os.chdir(home)

        sys.exit('Bulk analysis complete.\nUse QUBEKit -progress to view the completion progress of your molecules')


class Execute:
    """
    This class will be called for each INDIVIDUAL run and for each analysis in a BULK run.
    ArgsAndConfigs will handle bulk runs, calling this class as and when necessary.

    As such, this class needs to handle for each run:
        * Create and / or move into working dir
        * Create log file (Skipped if restarting)
        * Write to log file what is happening at the start of execution
            (starting or continuing; what are the configs; etc)
        * Execute any stages required, in the correct order according to self.order
        * Store info to log file as the program runs
            (which stages are being called; any key info / timings regarding progress; errors; etc)
        * Return results, both in individual stage directories and at the end
    """

    def __init__(self, molecule):

        # At this point, molecule should contain all of the config options from
        # the defaults, config file and terminal commands. These all come from the ArgsAndConfigs class.
        self.molecule = molecule

        self.start_up_msg = ('If QUBEKit ever breaks or you would like to view timings and loads of other info, '
                             'view the log file.\nOur documentation (README.md) '
                             'also contains help on handling the various commands for QUBEKit.\n')

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
                                  ('finalise', self.finalise),
                                  ('torsion_test', self.torsion_test)])

        # Keep this for pickling
        self.immutable_order = list(self.order)

        self.engine_dict = {'psi4': PSI4, 'g09': Gaussian, 'g16': Gaussian, 'onetep': ONETEP}

        printf(self.start_up_msg)

        # If restart is None, then the analysis has not been started previously
        self.create_log() if self.molecule.restart is None else self.continue_log()

        self.redefine_order()

        self.run()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def redefine_order(self):
        """
        If any order changes are required (restarting, new end point, skipping stages), it is done here.
        Creates a new self.order based on self.molecule's configs.
        """

        # If we are doing a torsion_test we have to redo the order as follows:
        # 1 Skip the finalise step to create a pickled ligand at the torsion_test stage
        # 2 Do the torsion_test and delete the torsion_test attribute
        # 3 Do finalise again to save the ligand with the correct attributes
        if hasattr(self.molecule, 'torsion_test'):
            self.order = OrderedDict([('finalise', self.skip), ('torsion_test', self.torsion_test), ('finalise', self.skip)])

        else:
            start = self.molecule.restart if self.molecule.restart is not None else 'parametrise'
            end = self.molecule.end if self.molecule.end is not None else 'finalise'
            skip = self.molecule.skip if self.molecule.skip is not None else []

            # Create list of all keys
            stages = list(self.order)

            # Cut out the keys before the start_point and after the end_point
            # Add finalise back in if it's removed (finalise should always be called).
            stages = stages[stages.index(start):stages.index(end) + 1] + ['finalise']

            # Redefine self.order to only contain the key, val pairs from stages
            self.order = OrderedDict(pair for pair in self.order.items() if pair[0] in set(stages))

            for pair in self.order.items():
                if pair[0] in skip:
                    self.order[pair[0]] = self.skip
                else:
                    self.order[pair[0]] = pair[1]

    def create_log(self):
        """
        Creates the working directory for the job as well as the log file.
        This log file is then extended when:
            - decorators.timer_logger wraps a called method;
            - helpers.append_to_log() is called;
            - helpers.pretty_print() is called with to_file set to True;
            - decorators.exception_logger() wraps a function / method which throws an exception.
                (This captures almost all errors)

        This method also makes backups if the working directory already exists.
        """

        date = datetime.now().strftime('%Y_%m_%d')

        # Define name of working directory.
        # This is formatted as 'QUBEKit_molecule name_yyyy_mm_dd_log_string'.

        dir_name = f'QUBEKit_{self.molecule.name}_{date}_{self.molecule.log}'

        # If you can't make a dir because it exists, back it up.
        try:
            os.mkdir(dir_name)

        except FileExistsError:
            # Try make a backup folder
            try:
                os.mkdir('QUBEKit_backups')
                printf('Making backup folder: QUBEKit_backups')
            except FileExistsError:
                # Backup folder already made
                pass
            # Keep increasing backup number until that particular number does not exist
            finally:
                count = 1
                while os.path.exists(f'QUBEKit_backups/{dir_name}_{str(count).zfill(3)}'):
                    count += 1

                # Then, make that backup and make a new working directory
                move(dir_name, f'QUBEKit_backups/{dir_name}_{str(count).zfill(3)}')
                printf(f'Moving directory: {dir_name} to backup folder')
                os.mkdir(dir_name)

        # Finally, having made any necessary backups, move files and change to working dir.
        finally:
            os.chdir(dir_name)

            # Set a home directory
            self.molecule.home = os.getcwd()

        # Find external files
        copy_files = [f'{self.molecule.name}.xml', self.molecule.filename]
        for file in copy_files:
            try:
                copy(f'../{file}', file)
            except FileNotFoundError:
                pass
            except TypeError:
                pass

        with open('QUBEKit_log.txt', 'w+') as log_file:
            log_file.write(f'Beginning log file; the time is: {datetime.now()}\n\n\n')

        self.log_configs()

    def continue_log(self):
        """
        In the event of restarting an analysis, find and append to the existing log file
        rather than creating a new one.
        """

        with open('QUBEKit_log.txt', 'a+') as log_file:
            log_file.write(f'\n\nContinuing log file from previous execution; the time is: {datetime.now()}\n\n\n')

        self.log_configs()

    def log_configs(self):
        """Writes the runtime and file-based defaults to a log file."""

        with open('QUBEKit_log.txt', 'a+') as log_file:

            log_file.write(f'Analysing: {self.molecule.name}\n\n')

            log_file.write('The runtime defaults and config options are as follows:\n\n')
            for key, val in self.molecule.__dict__.items():
                if val is not None:
                    log_file.write(f'{key}: {val}\n')
            log_file.write('\n')

    @exception_logger
    def run(self, torsion_options=None):
        # TODO bulk torsion options need to be checked
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order
            (according to self.order).
        Exceptions are added to log (if raised) using the decorator.
        """

        if 'parametrise' in self.order:
            if torsion_options is not None:
                torsion_options = torsion_options.split(',')
                self.molecule = self.store_torsions(self.molecule, torsion_options)
            self.molecule.pickle(state='parametrise')

        stage_dict = {
            'parametrise': ['Parametrising molecule', 'Molecule parametrised'],
            'mm_optimise': ['Partially optimising with MM', 'Partial optimisation complete'],
            'qm_optimise': ['Optimising molecule, view .xyz file for progress', 'Molecule optimisation complete'],
            'hessian': ['Calculating Hessian matrix', 'Hessian matrix calculated and confirmed to be symmetric'],
            'mod_sem': ['Calculating bonds and angles with modified Seminario method', 'Bonds and angles calculated'],
            'density': [f'Performing density calculation with {self.molecule.density_engine}',
                        'Density calculation complete'],
            'charges': [f'Chargemol calculating charges using DDEC{self.molecule.ddec_version}', 'Charges calculated'],
            'lennard_jones': ['Performing Lennard-Jones calculation', 'Lennard-Jones parameters calculated'],
            'torsion_scan': ['Performing QM-constrained optimisation with Torsiondrive',
                             'Torsiondrive finished and QM results saved'],
            'torsion_optimise': ['Performing torsion optimisation', 'Torsion optimisation complete'],
            'finalise': ['Finalising analysis', 'Molecule analysis complete!'],
            'pause': ['Pausing analysis', 'Analysis paused!'],
            'skip': ['Skipping section', 'Section skipped'],
            'torsion_test': ['Testing torsion single point energies', 'Torsion testing complete']}

        # Do the first stage in the order to get the next_key for the following loop
        key = list(self.order)[0]
        next_key = self.stage_wrapper(key, stage_dict[key][0], stage_dict[key][1], torsion_options)

        # Cannot use for loop as we mutate the dictionary during the loop
        while True:
            if next_key is None:
                break
            else:
                next_key = self.stage_wrapper(next_key, stage_dict[next_key][0], stage_dict[next_key][1])

            if next_key == 'pause':
                self.pause()
                break

    def stage_wrapper(self, start_key, begin_log_msg='', fin_log_msg='', torsion_options=None):
        """
        Firstly, check if the stage start_key is in self.order; this tells you if the stage should be called or not.
        If it isn't in self.order:
            - Do nothing
        If it is:
            - Unpickle the ligand object at the start_key stage
            - Write to the log that something's about to be done (if specified)
            - Make (if not restarting) and / or move into the working directory for that stage
            - Do the thing
            - Move back out of the working directory for that stage
            - Write to the log that something's been done (if specified)
            - Pickle the ligand object again with the next_key marker as its stage
        """

        mol = unpickle()[start_key]

        # Set the state for logging any exceptions should they arise
        mol.state = start_key

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

        folder_name = f'{str(self.immutable_order.index(start_key) + 1).zfill(2)}_{start_key}'
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

    @staticmethod
    def parametrise(molecule):
        """Perform initial molecule parametrisation using OpenFF, Antechamber or XML."""

        append_to_log('Starting parametrisation')

        # Parametrisation options:
        param_dict = {'antechamber': AnteChamber, 'xml': XML, 'openff': OpenFF, 'none': Parametrisation}

        # If we are using xml we have to move it
        if molecule.parameter_engine == 'xml':
            copy(os.path.join(molecule.home, f'{molecule.name}.xml'), f'{molecule.name}.xml')

        # Perform the parametrisation
        # If the method is none the molecule is not parameterised but the parameter holders are initiated
        if molecule.parameter_engine == 'none':
            param_dict[molecule.parameter_engine](molecule).gather_parameters()
        else:
            # Write the PDB file this covers us if we have a mol2 or xyz input file
            molecule.write_pdb()
            param_dict[molecule.parameter_engine](molecule)

        append_to_log(f'Finishing parametrisation of molecule with {molecule.parameter_engine}')

        return molecule

    @staticmethod
    def mm_optimise(molecule):
        """
        Use an mm force field to get the initial optimisation of a molecule

        options
        ---------
        RDKit MFF or UFF force fields can have strange effects on the geometry of molecules

        Geometric / OpenMM depends on the force field the molecule was parameterised with gaff/2, OPLS smirnoff.
        """

        append_to_log('Starting mm_optimisation')
        # Check which method we want then do the optimisation
        if molecule.mm_opt_method == 'openmm':
            if molecule.parameter_engine != 'none':
                # Make the inputs
                molecule.write_pdb(input_type='input')
                molecule.write_parameters()
                # Run geometric
                # TODO Should this be moved to allow a decorator?
                with open('log.txt', 'w+') as log:
                    sp.run(f'geometric-optimize --reset --epsilon 0.0 --maxiter {molecule.iterations} --pdb '
                           f'{molecule.name}.pdb --openmm {molecule.name}.xml '
                           f'{molecule.constraints_file if molecule.constraints_file is not None else ""}',
                           shell=True, stdout=log, stderr=log)

                # This will continue even if we don't converge this is fine
                # Read the xyz traj and store the frames
                molecule.read_xyz(f'{molecule.name}_optim.xyz')
                # Store the last from the traj as the mm optimised structure
                molecule.coords['mm'] = molecule.coords['traj'][-1]
            else:
                raise OptimisationFailed('You can not optimise a molecule with OpenMM and no initial parameters; '
                                         'consider parametrising or using UFF/MFF in RDKit')

        elif molecule.mm_opt_method == 'none':
            # Skip the optimisation step
            molecule.coords['mm'] = molecule.coords['input']

        else:
            # TODO change to qcengine as this can already be done
            # Run an rdkit optimisation with the right FF
            rdkit_ff = {'rdkit_mff': 'MFF', 'rdkit_uff': 'UFF'}
            molecule.filename = RDKit().mm_optimise(molecule.filename, ff=rdkit_ff[molecule.mm_opt_method])

        append_to_log(f'Finishing mm_optimisation of the molecule with {molecule.mm_opt_method}')

        return molecule

    def qm_optimise(self, molecule):
        """Optimise the molecule with or without geometric."""

        # TODO This method's not always printing completion to log file.
        # TODO If the optimisation restarts, we should print that to the log file.
        #   Also, restarts should restart from the partially optimised structure (last trajectory before failure)

        append_to_log('Starting qm_optimisation')
        qm_engine = self.engine_dict[molecule.bonds_engine](molecule)

        if molecule.geometric and molecule.bonds_engine == 'psi4':

            qceng = QCEngine(molecule)
            # See if the structure is there if not we did not optimise
            result = qceng.call_qcengine('geometric', 'gradient',
                                         input_type=f'{"mm" if list(molecule.coords["mm"]) else "input"}')

            # Check if converged and get the geometry
            if result['success']:

                # Load all of the frames into the molecule's trajectory holder
                molecule.read_geometric_traj(result['trajectory'])

                # store the final molecule as the qm optimised structure
                molecule.coords['qm'] = np.array(result['final_molecule']['geometry']).reshape((len(molecule.atoms), 3))
                molecule.coords['qm'] *= constants.BOHR_TO_ANGS

                molecule.qm_energy = result['energies'][-1]

                # Write out the trajectory file
                molecule.write_xyz(input_type='traj', name=f'{molecule.name}_opt')
                molecule.write_xyz(input_type='qm', name='opt')

                append_to_log(f'Finishing qm_optimisation of molecule{" using geometric" if molecule.geometric else ""}')

                return molecule

            else:
                restart_count = 1
                while not result['success'] and restart_count < 3:

                    # If the error was not a straight segfault se if we can get the molecule so far
                    try:
                        molecule.coords['temp'] = np.array(result['input_data']['final_molecule']['geometry']).reshape((len(molecule.atoms), 3))
                        molecule.coords['temp'] *= constants.BOHR_TO_ANGS

                        result = qceng.call_qcengine('geometric', 'gradient', 'temp')

                    except KeyError:
                        # No molecule data was caught so start again
                        # TODO check for other errors here as well
                        result = qceng.call_qcengine(
                            'geometric', 'gradient', f'{"mm" if list(molecule.coords["mm"]) else "input"}')

                if result['success']:
                    molecule.coords['qm'] = np.array(result['final_molecule']['geometry']).reshape((len(molecule.atoms), 3))
                    molecule.coords['qm'] *= constants.BOHR_TO_ANGS

                    molecule.qm_energy = result['energies'][-1]

                    return molecule

                else:
                    raise OptimisationFailed("The optimisation did not converge")

        result = qm_engine.generate_input(input_type=f'{"mm" if list(molecule.coords["mm"]) else "input"}',
                                          optimise=True, execute=molecule.bonds_engine)

        # Check the exit status of the job; if failed restart the job up to 2 times
        restart_count = 1
        while not result['success'] and restart_count < 3:
            append_to_log(f'{molecule.bonds_engine} optimisation failed with error {result["error"]}; restarting',
                          msg_type='minor')

            # 1) If we have a file read error just start again
            if result['error'] == 'FileIO':
                result = qm_engine.generate_input(input_type='mm', optimise=True, restart=True, execute=molecule.bonds_engine)

            # 2) If we have a distance matrix error we should start from a different structure try the input
            elif result['error'] == 'Distance matrix' and restart_count == 1:
                result = qm_engine.generate_input(input_type='input', optimise=True, execute=molecule.bonds_engine)

            # 3) If we have already tried the starting structure generate a conformer and try again
            elif result['error'] == 'Distance matrix':
                molecule.write_pdb()
                molecule.coords['temp'] = RDKit().generate_conformers(f'{molecule.name}.pdb')[0]
                result = qm_engine.generate_input(input_type='temp', optimise=True, execute=molecule.bonds_engine)

            # Some times the user has no given enough iterations so try again
            elif result['error'] == 'Max iterations':
                result = qm_engine.generate_input(input_type='input', optimise=True, restart=True, execute=self.molecule.bonds_engine)

            restart_count += 1

        if not result['success']:
            raise OptimisationFailed(f"{molecule.bonds_engine} "
                                     f"optimisation did not converge after 3 restarts; last error {result['error']}")

        molecule.coords['qm'], molecule.qm_energy = qm_engine.optimised_structure()
        molecule.write_xyz(input_type='qm', name='opt')

        append_to_log(f'Finishing qm_optimisation of molecule{" using geometric" if molecule.geometric else ""}')

        return molecule

    def hessian(self, molecule):
        """Using the assigned bonds engine, calculate and extract the Hessian matrix."""

        append_to_log('Starting hessian calculation')
        molecule.get_bond_lengths(input_type='qm')

        # Check what engine to use
        if molecule.bonds_engine == 'g09' or molecule.bonds_engine == 'g16':
            qm_engine = self.engine_dict[molecule.bonds_engine](molecule)

            # Use the checkpoint file as this has higher xyz precision
            try:
                copy(os.path.join(molecule.home, os.path.join('03_qm_optimise', 'lig.chk')), 'lig.chk')
                result = qm_engine.generate_input(input_type='qm', hessian=True, restart=True, execute=molecule.bonds_engine)
            except FileNotFoundError:
                append_to_log('qm_optimise checkpoint not found, optimising first to refine atomic coordinates')
                result = qm_engine.generate_input(input_type='qm', optimise=True, hessian=True, execute=molecule.bonds_engine)

            if result['success']:
                molecule.hessian = qm_engine.hessian()
            else:
                raise HessianCalculationFailed('The hessian was not calculated check the log file.')

        else:
            qceng = QCEngine(molecule)
            hessian = qceng.call_qcengine('psi4', 'hessian', input_type='qm')

            np.savetxt('hessian.txt', hessian)

            molecule.hessian = hessian

        append_to_log(f'Finishing Hessian calculation using {molecule.bonds_engine}')

        return molecule

    @staticmethod
    def mod_sem(molecule):
        """Modified Seminario for bonds and angles."""

        append_to_log('Starting mod_Seminario method')

        mod_sem = ModSeminario(molecule)
        mod_sem.modified_seminario_method()
        # Try and average out the new parameters
        molecule.symmetrise_bonded_parameters()

        append_to_log('Finishing Mod_Seminario method')

        return molecule

    def density(self, molecule):
        """Perform density calculation with the qm engine."""

        append_to_log('Starting density calculation')

        if molecule.density_engine == 'onetep':
            molecule.write_xyz(input_type='qm')
            # If using ONETEP, stop after this step
            append_to_log('Density analysis file made for ONETEP')

            # Edit the order to end here
            self.order = OrderedDict([('density', self.density), ('charges', self.skip), ('lennard_jones', self.skip),
                                      ('torsion_scan', self.torsion_scan), ('pause', self.pause)])

        else:
            qm_engine = self.engine_dict[molecule.density_engine](molecule)
            qm_engine.generate_input(input_type='qm', density=True, solvent=molecule.solvent, execute=molecule.density_engine)
            append_to_log('Finishing Density calculation')

        return molecule

    @staticmethod
    def charges(molecule):
        """Perform DDEC calculation with Chargemol."""

        append_to_log('Starting charge partitioning')
        copy(os.path.join(molecule.home, os.path.join('06_density', f'{molecule.name}.wfx')), f'{molecule.name}.wfx')
        c_mol = Chargemol(molecule)
        c_mol.generate_input()

        append_to_log(f'Finishing Charge partitioning with Chargemol and DDEC{molecule.ddec_version}')

        return molecule

    @staticmethod
    def lennard_jones(molecule):
        """Calculate Lennard-Jones parameters, and extract virtual sites."""

        append_to_log('Starting Lennard-Jones parameter calculation')

        charges_folder = os.path.join(molecule.home, '07_charges')
        for file in os.listdir(charges_folder):
            if file.startswith('DDEC'):
                copy(os.path.join(charges_folder, file), file)

        lj = LennardJones(molecule)
        molecule.NonbondedForce = lj.calculate_non_bonded_force()

        # This also now implies the opls combination rule
        molecule.combination = 'opls'

        append_to_log('Finishing Lennard-Jones parameter calculation')

        return molecule

    @staticmethod
    def torsion_scan(molecule):
        """Perform torsion scan."""
        append_to_log('Starting torsion_scans')

        scan = TorsionScan(molecule)
        # Check that we have a scan order for the molecule this should of been captured from the dihedral file
        scan.find_scan_order()

        scan.scan()

        append_to_log('Finishing torsion_scans')

        return molecule

    @staticmethod
    def torsion_optimise(molecule):
        """Perform torsion optimisation."""

        append_to_log('Starting torsion_optimisations')

        # First we should make sure we have collected the results of the scans
        if molecule.qm_scans is not None:
            os.chdir(os.path.join(molecule.home, '09_torsion_scan'))
            scan = TorsionScan(molecule)
            scan.find_scan_order()
            scan.collect_scan()
            os.chdir(os.path.join(molecule.home, '10_torsion_optimise'))

        opt = TorsionOptimiser(molecule, refinement=molecule.refinement_method, vn_bounds=molecule.tor_limit)
        opt.run()

        append_to_log('Finishing torsion_optimisations')

        return molecule

    @staticmethod
    def finalise(molecule):
        """
        Make the xml and pdb file;
        print the ligand object to terminal (in abbreviated form) and to the log file (unabbreviated).
        """

        molecule.write_pdb()
        molecule.write_parameters()

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

        printf('QUBEKit stopping at ONETEP step!\n To continue please move the ddec.onetep file and xyz file to the '
               'density folder and use QUBEKit -restart lennard_jones.')

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
                # convert the string names to the index
                core = (molecule.get_atom_with_name(tor[1]).atom_index, molecule.get_atom_with_name(tor[2]).atom_index)

                if core in molecule.rotatable:
                    scan_order.append(core)
                elif reversed(core) in molecule.rotatable:
                    scan_order.append(reversed(core))

            molecule.scan_order = scan_order

        return molecule

    @staticmethod
    def torsion_test(molecule):
        """Take the molecule and do the torsion test method."""

        # If there is a constraints file we should move it
        if molecule.constraints_file is not None:
            copy(molecule.constraints_file, molecule.constraints_file.name)

        # Now we should run the torsion_test method
        opt = TorsionOptimiser(molecule, refinement=molecule.refinement_method,
                               vn_bounds=molecule.tor_limit)
        opt.torsion_test()

        printf('Torsion testing done!')

        # Now we must remove the torsion_test attrib
        del molecule.torsion_test

        return molecule


def main():
    """Putting class call inside a function squashes the __repr__ call"""
    ArgsAndConfigs()


if __name__ == '__main__':
    # For running with debugger, normal entry point is defined in setup.py
    main()
