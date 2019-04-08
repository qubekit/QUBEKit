#!/usr/bin/env python

from QUBEKit.smiles import smiles_to_pdb, smiles_mm_optimise, rdkit_descriptors
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.engines import PSI4, Chargemol, Gaussian, ONETEP
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
from QUBEKit.parametrisation import OpenFF, AnteChamber, XML
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, append_to_log, pretty_progress, pretty_print, \
    Configure, unpickle
from QUBEKit.decorators import exception_logger_decorator

import argparse
from sys import exit as sys_exit
from os import mkdir, chdir, path, listdir, walk, getcwd, system
from shutil import copy
from collections import OrderedDict
from functools import partial
from datetime import datetime

# Changes default print behaviour for this file.
print = partial(print, flush=True)


class Main:
    """
    Interprets commands from the terminal.
    Stores defaults or executes relevant functions.
    Will also create log and working directory where needed.
    See README.md for detailed discussion of QUBEKit commands or method docstrings for specifics of their functionality.
    """

    def __init__(self):
        self.file = None
        self.start_up_msg = ('If QUBEKit ever breaks or you would like to view timings and loads of other info, '
                             'view the log file.\n Our documentation (README.md) '
                             'also contains help on handling the various commands for QUBEKit\n')

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
                                  ('torsion_optimisation', self.torsion_optimisation),
                                  ('finalise', self.finalise)])

        self.log_file = 'QUBEKit_log.txt'
        self.engine_dict = {'psi4': PSI4, 'g09': Gaussian, 'onetep': ONETEP}

        # Argparse will only return if we are doing a QUBEKit run bulk or normal
        self.args = self.parse_commands()

        # Look through the command line options and apply bulk and restart settings
        self.check_options()

        # Configs:
        self.defaults_dict = {'charge': self.args.charge,
                              'multiplicity': self.args.multiplicity,
                              'config': self.args.config_file}

        self.configs = {'qm': {},
                        'fitting': {},
                        'descriptions': {}}

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
        print(self.start_up_msg)
        self.execute_new()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def check_options(self):
        """Read through the command line and handle options which affect the main execute
        like restart or start a bulk run."""

        # Run the bulk command and exit the code on completion
        if self.args.bulk_run:
            self.bulk_execute()

        # If not bulk must be main single run
        self.file = self.args.input
        # Check the end points for a normal run
        start_point = self.args.restart if self.args.restart is not None else 'parametrise'
        end_point = self.args.end if self.args.end is not None else 'finalise'

        # Create list of all keys
        stages = [key for key in self.order]

        # This ensures that the run is start_point to end_point inclusive rather than exclusive.
        # e.g. -restart parametrise charges goes from parametrise to charges while doing the charges step.
        extra = 1 if end_point != 'finalise' else 0

        # Cut out the keys before the start_point and after the end_point
        # Add finalise back in if it's removed (finalise should always be called).
        stages = stages[stages.index(start_point):stages.index(end_point) + extra] + ['finalise']

        # Redefine self.order to only contain the key, val pairs from stages
        self.order = OrderedDict(pair for pair in self.order.items() if pair[0] in set(stages))

        # If we restart we should have a pickle file with the object in it.
        # if '-restart' in cmd:
        #     # Set the file name based on the directory:
        #     files = [file for file in listdir('.') if path.isfile(file)]
        #     self.file = \
        #         [file for file in files if file.endswith('.pdb') and not file.endswith('optimised.pdb')][0]

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
            - Four: look for smiles codes and input files

        This ordering ensures that the program will always:
            - Terminate immediately if necessary;
            - Store all config changes correctly before running anything;
            - Perform bulk analysis rather than single molecule analysis (if requested);
            - Restart or end in the correct way (if requested);
            - Perform single molecule analysis using a pdb or smiles string (if requested).

        After all commands have been parsed and appropriately used, either:
            - Commands are returned, along with the relevant file name
            - The program exits
            - return None
        """
        # action classes
        class SetupAction(argparse.Action):
            """The setup action class that is called when setup is found in the command line."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when setup is called."""
                choice = input('You can now edit config files using QUBEKit, choose an option to continue:\n'
                               '1) Edit a config file\n'
                               '2) Create a new master template\n'
                               '3) Make a normal config file\n>')

                if int(choice) == 1:
                    inis = Configure.show_ini()
                    name = input(
                        f'Enter the name or number of the config file to edit\n'
                        f'{"".join(f"{inis.index(ini)}:{ini}    " for ini in inis)}\n>')
                    # make sure name is right
                    if name in inis:
                        Configure.ini_edit(name)
                    else:
                        Configure.ini_edit(inis[int(name)])

                elif int(choice) == 2:
                    Configure.ini_writer('master_config.ini')
                    Configure.ini_edit('master_config.ini')

                elif int(choice) == 3:
                    name = input('Enter the name of the config file to create\n>')
                    Configure.ini_writer(name)
                    Configure.ini_edit(name)

                else:
                    raise KeyError('Invalid selection; please choose from 1, 2 or 3.')

                sys_exit()

        class CsvAction(argparse.Action):
            """The csv creation class run when the csv option is used."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when csv is called."""
                generate_config_csv(values)
                sys_exit()

        class ProgressAction(argparse.Action):
            """Run the pretty progress function to get the progress of all running jobs."""

            def __call__(self, pars, namespace, values, option_string=None):
                """This function is executed when progress is called."""
                pretty_progress()
                sys_exit()

        parser = argparse.ArgumentParser(prog='QUBEKit', formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="""QUBEKit is a Python 3.5+ based force field derivation toolkit for Linux operating systems.
Our aims are to allow users to quickly derive molecular mechanics parameters directly from quantum mechanical calculations.
QUBEKit pulls together multiple pre-existing engines, as well as bespoke methods to produce accurate results with minimal user input.
QUBEKit aims to use as few parameters as possible while also being highly customisable.""", epilog='''QUBEKit should currently be considered a work in progress.
While it is stable we are constantly working to improve the code and broaden its compatibility. 
We use lots of software written by many different people;
if reporting a bug please (to the best of your ability) make sure it is a bug with QUBEKit and not with a dependency.
We welcome any suggestions for additions or changes.''')

        # Add all of the command line options in the arg parser
        parser.add_argument('-c', '--charge', default=0, type=int, help='Enter the charge of the molecule, default 0.')
        parser.add_argument('-m', '--multiplicity', default=1, type=int, help='Enter the multiplicity of the '
                                                                              'molecule, default 1.')
        parser.add_argument('-ddec', '--ddec_version', choices=[3, 6], type=int,
                            help='Enter the ddec version for charge partitioning, does not effect ONETEP partitioning.')
        parser.add_argument('-geo', '--geometric', choices=[True, False], type=bool,
                            help='Turn on geometric to use this during the qm optimisations, recomended.')
        parser.add_argument('-bonds', '--bonds_engine', choices=['psi4', 'g09'],
                            help='Choose the QM code to calculate the bonded terms.')
        parser.add_argument('-charges', '--charges_engine', choices=['onetep', 'chargemol'],
                            help='Choose the method to do the charge partioning.')
        parser.add_argument('-density', '--density_engine', choices=['onetep', 'g09', 'psi4'],
                            help='Enter the name of the QM code to calculate the electron density of the molecule.')
        parser.add_argument('-solvent', '--solvent',
                            help='Enter the dielectric constant or the name of the solvent you wish to use.')
        # maybe separate into known solvents and IPCM constants?
        parser.add_argument('-convergence', '--convergence', choices=['GAU', 'GAU_TIGHT', 'GAU_VERYTIGHT'],
                            help='Enter the convergence criteria for the optimisation.')
        parser.add_argument('-param', '--parameter_engine', choices=['xml', 'gaff', 'gaff2', 'openff'],
                            help='Enter the method of where we should get the intial molecule parameters from, '
                                 'if xml make sure the xml has the same name as the pdb file.')
        parser.add_argument('-mm', '--mm_opt_method', default='openmm', choices=['openmm', 'rdkit_mff', 'rdkit_uff'],
                            help='Enter the mm optimisation method for pre qm omptimisation.')
        parser.add_argument('-config', '--config_file', default='default_config', choices=Configure.show_ini(),
                            help='Enter the name of the configuration file you wish to use for this run from the list '
                                 'available, defaults to master.')
        parser.add_argument('-theory', '--theory',
                            help='Enter the name of the qm theory you would like to use.')
        parser.add_argument('-basis', '--basis',
                            help='Enter the basis set you would like to use.')
        parser.add_argument('-restart', '--restart', choices=['parametrise', 'mm_optimise', 'qm_optimise', 'hessian',
                                                              'mod_sem', 'density', 'charges', 'lennard_jones',
                                                              'torsion_scan', 'torsion_optimisation'],
                            help='Enter the restart point of a QUBEKit job.')
        parser.add_argument('-end', '-end', choices=['mm_optimise', 'qm_optimise', 'hessian', 'mod_sem', 'density',
                                                     'charges', 'lennard_jones', 'torsion_scan', 'torsion_optimisation',
                                                     'finalise'], help='Enter the end point of the QUBEKit job.')
        parser.add_argument('-progress', '--progress', nargs='?', const=True,
                            help='Get the current progress of a QUBEKit single or bulk job.', action=ProgressAction)
        parser.add_argument('-combination', '--combination', default='opls', choices=['opls', 'amber'],
                            help='Enter the combination rules that should be used.')

        # Add mutually exclusive groups to stop wrong combinations of options,
        # e.g. setup should not be ran with another command
        groupa = parser.add_mutually_exclusive_group()
        groupa.add_argument('-setup', '--setup_config', nargs='?', const=True,
                            help='Setup a new configuration or edit an existing one.', action=SetupAction)
        groupa.add_argument('-sm', '--smiles', help='Enter the smiles string of a molecule as a starting point.')
        groupa.add_argument('-bulk', '--bulk_run',
                            help='Enter the name of the csv file to run as bulk, bulk will use smiles unless it finds '
                                 'a molecule file with the same name.')
        groupa.add_argument('-csv', '--csv_filename',
                            help='Enter the name of the csv file you would like to create for bulk runs.',
                            action=CsvAction)
        groupa.add_argument('-i', '--input', help='Enter the molecule input pdb file (only pdb so far!)')

        return parser.parse_args()

    def bulk_execute(self):
        """Run a bulk QUBEKit job in serial mode."""
        # TODO look at worker queues to maximise resource usage

        csv_file = self.args.bulk_run
        print(self.start_up_msg)

        bulk_data = get_mol_data_from_csv(csv_file)

        # Run full analysis for each smiles string or pdb in the .csv file.
        names = list(bulk_data.keys())
        # Store a copy of self.order which will not be mutated.
        # This allows self.order to be built up after each run.
        temp = self.order

        for name in names:

            print(f'Currently analysing: {name}\n')

            # Set the start and end points to what is given in the csv. See the -restart / -end section below
            # for further details and better documentation.
            start_point = bulk_data[name]['start'] if bulk_data[name]['start'] else 'parametrise'
            end_point = bulk_data[name]['end']
            stages = [key for key in temp]
            extra = 1 if end_point != 'finalise' else 0
            stages = stages[stages.index(start_point):stages.index(end_point) + extra] + ['finalise']
            self.order = OrderedDict(pair for pair in temp.items() if pair[0] in set(stages))

            # Configs
            self.defaults_dict = bulk_data[name]
            self.qm, self.fitting, self.descriptions = Configure.load_config(self.defaults_dict['config'])
            self.all_configs = [self.defaults_dict, self.qm, self.fitting, self.descriptions]

            # If starting from the beginning, create log and pdb file then execute as normal for each run
            if start_point == 'parametrise':

                if bulk_data[name]['smiles string'] is not None:
                    smile_string = bulk_data[name]['smiles string']
                    self.file = smiles_to_pdb(smile_string, name)

                else:
                    self.file = f'{name}.pdb'

                self.create_log()

            # If starting from the middle somewhere, FIND (not create) the folder, and log and pdb files, then execute
            else:
                for root, dirs, files in walk('.', topdown=True):
                    for dir_name in dirs:
                        if dir_name.startswith(f'QUBEKit_{name}'):
                            chdir(dir_name)

                # These are the files in the active directory, search for the pdb.
                files = [file for file in listdir('.') if path.isfile(file)]
                self.file = \
                    [file for file in files if file.endswith('.pdb') and not file.endswith('optimised.pdb')][0]

                self.continue_log()

            self.execute_new()
            chdir('../')

        sys_exit('\nFinished bulk run. Use the command -progress to view which stages have completed.')

    def continue_log(self):
        """
        In the event of restarting an analysis, find and append to the existing log file
        rather than creating a new one.
        """

        with open(self.log_file, 'a+') as log_file:

            log_file.write(f'\n\nContinuing log file from previous execution: {datetime.now()}\n\n')
            log_file.write(f'The commands given were: {f"{key}: {value}" for key, value in vars(self.args).items() if value is not None}\n\n')

            # TODO Add logic to reprint commands with *s after changed defaults.
            #   Could possibly be done using the pickle file? Are the configs stored in an usable / accessible form?
            # Writes the config dictionaries to the log file.
            log_file.write('The defaults being used are:\n')
            for dic in self.all_configs:
                for key, var in dic.items():
                    log_file.write(f'{key}: {var}\n')
                log_file.write('\n')

            log_file.write('\n')

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
        dir_string = f'QUBEKit_{self.file[:-4]}_{date}_{self.descriptions["log"]}'
        mkdir(dir_string)

        # Copy active pdb into new directory.
        abspath = path.abspath(self.file)
        copy(abspath, f'{dir_string}/{self.file}')
        chdir(dir_string)

        with open(self.log_file, 'w+') as log_file:

            log_file.write(f'Beginning log file: {datetime.now()}\n\n')
            log_file.write(f'The commands given were: {f"{key}: {value}" for key, value in vars(self.args).items() if value is not None}\n\n')
            log_file.write(f'Analysing: {self.file[:-4]}\n\n')

            # Writes the config dictionaries to the log file.
            log_file.write('The defaults being used are:\n')
            for config in self.all_configs:
                for key, var in config.items():
                    log_file.write(f'{key}: {var}\n')
                log_file.write('\n')

            log_file.write('\n')

    def stage_wrapper(self, start_key, begin_log_msg='', fin_log_msg=''):
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

        # if start_key in [key for key in self.order]:

        mol = unpickle(f'.{self.file[:-4]}_states')[start_key]

        if begin_log_msg:
            print(f'{begin_log_msg}...', end=' ')

        # move into folder
        home = getcwd()
        try:
            mkdir(f'{start_key}')
        except FileExistsError:
            pass
        finally:
            chdir(f'{start_key}')
        self.order[start_key](mol)
        self.order.pop(start_key, None)
        chdir(home)
        # Begin looping through self.order, but return after the first iteration.
        for key in self.order:
            next_key = key
            if fin_log_msg:
                print(fin_log_msg)

            #mol.pickle(state=self.order[next_key].__name__)
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
            copy(f'../../{molecule.name}.xml', f'{molecule.name}.xml')

        # Perform the parametrisation
        param_dict[self.fitting['parameter_engine']](molecule)

        append_to_log(f'Parametrised molecule with {self.fitting["parameter_engine"]}')

        return molecule

    def mm_optimise(self, molecule):
        """Use a mm force field to get the initial optimisation of a molecule

        options
        ---------
        RDKit MFF or UFF force fields can have strange effects on the geometry of molecules

        Geometric/ OpenMM depends on the force field the molecule was parameterised with gaff/2, OPLS smirnoff.
        """

        # Check which method we want then do the optimisation
        if self.args.mm_opt_method == 'openmm':
            # make the inputs
            molecule.write_pdb(name='openmm', input_type='input')
            molecule.write_parameters(name='input')
            # run geometric
            system('geometric-optimize --reset --epsilon 0.0 --maxiter 500 --qccnv --openmm openmm.pdb > log.txt')
            # get the optimised structure store under mm
            molecule.read_xyz(input_type='mm')

        else:
            # run a rdkit optimisation with the right FF
            rdkit_ff = {'rdkit_mff': 'MFF', 'rdkit_uff': 'UFF'}
            molecule.filename = smiles_mm_optimise(molecule.filename, ff=rdkit_ff[self.args.mm_opt_method])

        append_to_log(f'Optimised the molecule with {self.args.mm_opt_method}')

        return molecule

    def qm_optimise(self, molecule):
        """Optimise the molecule with or without geometric."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)

        if self.qm['geometric']:

            # Calc geometric-related gradient and geometry
            qm_engine.geo_gradient(input_type='mm')
            molecule.read_xyz(input_type='qm')

        else:
            qm_engine.generate_input(input_type='mm', optimise=True)
            molecule.molecule['qm'] = qm_engine.optimised_structure()

        append_to_log(f'Optimised structure calculated{" with geometric" if self.qm["geometric"] else ""}')

        return molecule

    def hessian(self, molecule):
        """Using the assigned bonds engine, calculate and extract the Hessian matrix."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)

        # Write input file for bonds engine
        qm_engine.generate_input(input_type='qm', hessian=True)

        # Calc bond lengths from molecule topology
        molecule.get_bond_lengths(input_type='qm')

        # Extract Hessian and modes
        molecule.hessian = qm_engine.hessian()
        molecule.modes = qm_engine.all_modes()

        return molecule

    def mod_sem(self, molecule):
        """Modified Seminario for bonds and angles."""

        mod_sem = ModSeminario(molecule, self.all_configs)
        mod_sem.modified_seminario_method()

        append_to_log('Modified Seminario method complete')

        return molecule

    def density(self, molecule):
        """Perform density calculation with the qm engine."""

        qm_engine = self.engine_dict[self.qm['density_engine']](molecule, self.all_configs)
        qm_engine.generate_input(input_type='qm', density=True, solvent=self.qm['solvent'])

        if self.qm['density_engine'] == 'g09':
            append_to_log('Gaussian analysis complete')
        else:
            # If we use onetep we have to stop after this step
            append_to_log('ONETEP file made')

            # Now we have to edit the order to end here.
            self.order = OrderedDict([('density', self.density), ('charges', self.skip), ('lennard_jones', self.skip),
                                      ('torsion_scan', self.torsion_scan), ('pause', self.pause)])

        return molecule

    def charges(self, molecule):
        """Perform DDEC calculation with Chargemol."""

        # TODO add option to use chargemol on onetep cube files.
        copy(f'../density/{molecule.name}.wfx', f'{molecule.name}.wfx')
        c_mol = Chargemol(molecule, self.all_configs)
        c_mol.generate_input()

        append_to_log(f'Chargemol analysis with DDEC{self.qm["ddec_version"]} complete')

        return molecule

    def lennard_jones(self, molecule):
        """Calculate Lennard-Jones parameters, and extract virtual sites."""

        system('cp ../charges/DDEC* .')
        lj = LennardJones(molecule, self.all_configs)
        molecule.NonbondedForce = lj.calculate_non_bonded_force()

        append_to_log('Lennard-Jones parameters calculated')

        return molecule

    def torsion_scan(self, molecule):
        """Perform torsion scan."""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        scan = TorsionScan(molecule, qm_engine, self.all_configs)
        scan.start_scan()

        append_to_log('Torsion scans complete')

        return molecule

    def torsion_optimisation(self, molecule):
        """Perform torsion optimisation"""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        opt = TorsionOptimiser(molecule, qm_engine, self.all_configs, opt_method='BFGS', combination=self.args.combination,
                               refinement_method=self.args.refinement_method,
                               vn_bounds=self.args.tor_limit)
        opt.run()

        append_to_log('Torsion optimisations complete')

        return molecule

    @staticmethod
    def finalise(molecule):
        """Make the xml and pdb file print the ligand object to terminal (in abbreviated form) and to the log file
        after getting the rdkit descriptors.
        """

        # write the pdb file and xml file to the folder
        molecule.write_pdb()
        molecule.write_parameters()

        # get the molecule descriptors from rdkit
        molecule.descriptors = rdkit_descriptors(molecule.filename)

        # Print ligand objects to log file and terminal
        pretty_print(molecule, to_file=True)
        pretty_print(molecule)

        return molecule

    @staticmethod
    def skip(molecule):
        """
        A blank method that does nothing to that stage but adds the pickle points to not break the flow
        """

        return molecule

    @staticmethod
    def pause():
        """
        Pause the analysis when using onetep so we can comeback into the work flow but do not edit the pickling process
        """

        print('QUBEKit stopping at onetep step!\n To continue please move the ddec.onetep file and xyz file to the '
              'density folder and use -restart lennard_jones to continue.')

        sys_exit()


    @exception_logger_decorator
    def execute(self):
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order.
        Exceptions are added to log (if raised).
        """

        # Check if starting from the beginning; if so:
        if 'parametrise' in [key for key in self.order]:
            # Initialise ligand object fully before pickling it
            molecule = Ligand(self.file, combination=self.args.combination)
            molecule.pickle(state='parametrise')

        # Perform each key stage sequentially adding short messages (if given) to terminal to show progress.
        # Longer messages should be written inside the key stages' functions using helpers.append_to_log().
        # See PSI4 class in engines for an example of where this is used.
        self.stage_wrapper('parametrise', 'Parametrising molecule', 'Molecule parametrised')
        self.stage_wrapper('mm_optimise', 'Partially optimising with MM', 'Optimisation complete')
        self.stage_wrapper('qm_optimise', 'Optimising molecule, view .xyz file for progress', 'Molecule optimised')
        self.stage_wrapper('hessian', 'Calculating Hessian matrix')
        self.stage_wrapper('mod_sem', 'Calculating bonds and angles with modified Seminario method',
                           'Bonds and angles calculated')
        self.stage_wrapper('density', f'Performing density calculation with {self.args.density_engine}',
                           'Density calculation complete')
        self.stage_wrapper('charges', f'Chargemol calculating charges using DDEC{self.qm["ddec_version"]}',
                           'Charges calculated')
        self.stage_wrapper('lennard_jones', 'Performing Lennard-Jones calculation',
                           'Lennard-Jones parameters calculated')
        self.stage_wrapper('torsion_scan', 'Performing QM constrained optimisation with torsiondrive',
                           'Torsiondrive finished QM results saved')
        self.stage_wrapper('torsion_optimisation', 'Performing torsion optimisation', 'Torsion optimisation complete')

        # This step is always performed
        self.stage_wrapper('finalise', 'Finalising analysis', 'Molecule analysis complete!')

        # This step is only performed if we need to use onetep hence we have to pause the flow
        self.stage_wrapper('pause', 'Analysis stopping', 'Analysis paused!')

    @exception_logger_decorator
    def execute_new(self):
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order.
        Exceptions are added to log (if raised).
        """

        # Check if starting from the beginning; if so:
        if 'parametrise' in [key for key in self.order]:
            # Initialise ligand object fully before pickling it
            molecule = Ligand(self.file, combination=self.args.combination)
            molecule.pickle(state='parametrise')

        # Perform each key stage sequentially adding short messages (if given) to terminal to show progress.
        # Longer messages should be written inside the key stages' functions using helpers.append_to_log().
        # See PSI4 class in engines for an example of where this is used.

        # The stage keys and messages
        stage_dict = {'parametrise': ['Parametrising molecule', 'Molecule parametrised'],
                      'mm_optimise': ['Partially optimising with MM', 'Optimisation complete'],
                      'qm_optimise': ['Optimising molecule, view .xyz file for progress', 'Molecule optimised'],
                      'hessian': ['Calculating Hessian matrix', 'Matrix calculated'],
                      'mod_sem': ['Calculating bonds and angles with modified Seminario method',
                                  'Bonds and angles calculated'],
                      'density': [f'Performing density calculation with {self.qm["density_engine"]}',
                                  'Density calculation complete'],
                      'charges': [f'Chargemol calculating charges using DDEC{self.qm["ddec_version"]}',
                                  'Charges calculated'],
                      'lennard_jones': ['Performing Lennard-Jones calculation', 'Lennard-Jones parameters calculated'],
                      'torsion_scan': ['Performing QM constrained optimisation with torsiondrive',
                                       'Torsiondrive finished QM results saved'],
                      'torsion_optimisation': ['Performing torsion optimisation', 'Torsion optimisation complete'],
                      'finalise': ['Finalising analysis', 'Molecule analysis complete!'],
                      'pause': ['Analysis stopping', 'Analysis paused!'],
                      'skip': ['Skipping section', 'Section skipped']}

        # do the first stage in the order
        key = list(self.order.keys())[0]
        next_key = self.stage_wrapper(key, stage_dict[key][0], stage_dict[key][1])

        # cannot use for loop as we mute the dictionary during the loop
        while True:
            next_key = self.stage_wrapper(next_key, stage_dict[next_key][0], stage_dict[next_key][1])

            if next_key == 'pause':
                self.pause()




def main():
    """This just stops the __repr__ automatically being called when the class is called."""
    Main()
