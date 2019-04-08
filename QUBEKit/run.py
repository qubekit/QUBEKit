#!/usr/bin/env python

from QUBEKit.smiles import smiles_to_pdb, smiles_mm_optimise
from QUBEKit.mod_seminario import ModSeminario
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.engines import PSI4, Chargemol, Gaussian, ONETEP
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan, TorsionOptimiser
from QUBEKit.parametrisation import OpenFF, AnteChamber, XML
from QUBEKit.helpers import get_mol_data_from_csv, generate_config_csv, append_to_log, pretty_progress, pretty_print, Configure, unpickle
from QUBEKit.decorators import exception_logger_decorator

from sys import argv as cmdline
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

        self.start_up_msg = ('If QUBEKit ever breaks or you would like to view timings and loads of other info, '
                             'view the log file.\n Our documentation (README.md) '
                             'also contains help on handling the various commands for QUBEKit\n')

        # Configs:
        self.defaults_dict = {'charge': 0, 'multiplicity': 1, 'config': 'default_config'}
        self.configs = {'qm': {}, 'fitting': {}, 'descriptions': {}}

        # Call order of the analysing methods.
        # Slices of this dict are taken when changing the start and end points of analyses.
        self.order = OrderedDict([('rdkit_optimise', self.rdkit_optimise),
                                  ('parametrise', self.parametrise),
                                  ('qm_optimise', self.qm_optimise),
                                  ('hessian', self.hessian),
                                  ('mod_sem', self.mod_sem),
                                  ('density', self.density),
                                  ('charges', self.charges),
                                  ('lennard_jones', self.lennard_jones),
                                  ('torsion_scan', self.torsion_scan),
                                  ('torsion_optimise', self.torsion_optimise),
                                  ('finalise', self.finalise)])

        self.engine_dict = {'psi4': PSI4, 'g09': Gaussian, 'onetep': ONETEP}

        self.log_file = 'QUBEKit_log.txt'
        self.file, self.commands = self.parse_commands()

        # Find which config is being used and store arguments accordingly
        if self.defaults_dict['config'] == 'default_config':
            if not Configure.check_master():
                # Press any key to continue
                input('You must set up a master config to use QUBEKit and change the chargemol path; '
                      'press enter to edit master config. \n'
                      'You are free to change it later, with whichever editor you prefer.')
                Configure.ini_writer('master_config.ini')
                Configure.ini_edit('master_config.ini')

        self.qm, self.fitting, self.descriptions = Configure.load_config(self.defaults_dict['config'])
        self.all_configs = [self.defaults_dict, self.qm, self.fitting, self.descriptions]

        self.config_update()
        self.continue_log() if '-restart' in self.commands else self.create_log()
        self.execute()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def config_update(self):
        """Update the config settings with the command line ones from parse_commands()."""

        for key in self.configs:
            for sub in self.configs[key]:
                if sub in self.qm:
                    self.qm[sub] = self.configs[key][sub]
                elif sub in self.fitting:
                    self.fitting[sub] = self.configs[key][sub]
                elif sub in self.descriptions:
                    self.descriptions[sub] = self.configs[key][sub]

    def parse_commands(self):
        """
        Parses commands from the terminal.

        This method has four main blocks, each defined by an enumerate loop which loops over the commands.
        In the first block:
            - First, the program will search for commands which terminate the program such as -progress, -setup.
            - Then, it will search for, and change, config settings such as -c, -ddec.
        In the second block:
            - Search for, and execute bulk commands
        In the third block:
            - Search for, and execute -restart and -end commands
        In the fourth and final block:
            - Search for, and execute -sm commands or plain old named .pdb files.

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

        self.commands = [item.lower() for item in cmdline[1:]]

        # Check for csv generation, progress displaying or config setup first. These will halt the entire program.
        # Then check for config changes; multiple configs can be changed at once.
        for count, cmd in enumerate(self.commands):

            # csv creation
            if cmd == '-csv':
                csv_name = self.commands[count + 1]
                generate_config_csv(csv_name)
                sys_exit()

            # Display progress of all folders in current directory.
            if cmd == '-progress':
                pretty_progress()
                sys_exit()

            # Setup configs for all future runs
            if cmd == '-setup':
                choice = int(input('You can now edit config files using QUBEKit, choose an option to continue:\n'
                                   '1) Edit a config file\n'
                                   '2) Create a new master template\n'
                                   '3) Make a normal config file\n>'))

                if choice == 1:
                    inis = Configure.show_ini()
                    name = input(f'Enter the name of the config file to edit\n{"".join(f"{ini}    " for ini in inis)}\n>')
                    Configure.ini_edit(name)

                elif choice == 2:
                    Configure.ini_writer('master_config.ini')
                    Configure.ini_edit('master_config.ini')

                elif choice == 3:
                    name = input('Enter the name of the config file to create\n>')
                    Configure.ini_writer(name)
                    Configure.ini_edit(name)

                else:
                    raise KeyError('Invalid selection; please choose from 1, 2 or 3.')

                sys_exit()

            if cmd == '-c':
                self.defaults_dict['charge'] = int(self.commands[count + 1])

            if cmd == '-m':
                self.defaults_dict['multiplicity'] = int(self.commands[count + 1])

            if cmd == '-ddec':
                self.configs['qm']['ddec_version'] = int(self.commands[count + 1])

            if cmd == '-geo':
                self.configs['qm']['geometric'] = False if self.commands[count + 1] == 'false' else True

            if cmd == '-bonds':
                self.configs['qm']['bonds_engine'] = str(self.commands[count + 1])

            if cmd == '-charges':
                self.configs['qm']['charges_engine'] = str(self.commands[count + 1])

            if cmd == '-log':
                self.configs['descriptions']['log'] = str(self.commands[count + 1])

            if cmd == '-solvent':
                self.configs['qm']['solvent'] = False if self.commands[count + 1] == 'false' else True

            if cmd == '-param':
                self.configs['fitting']['parameter_engine'] = str(self.commands[count + 1])

            # Unlike '-setup', this just changes the config file used for this particular run.
            if cmd == '-config':
                self.defaults_dict['config'] = str(self.commands[count + 1])

            if cmd == '-func':
                self.configs['qm']['theory'] = str(self.commands[count + 1])

            if cmd == '-basis':
                self.configs['qm']['basis'] = str(self.commands[count + 1])

        if self.commands:
            print(f'\nThese are the commands you gave: {self.commands}\n' 
                  f'These are the current defaults: {self.defaults_dict} \n\n'
                  'Please note, some values may not be used depending on what kind of analysis is being done.\n')

        # Check if a bulk analysis is being done.
        for count, cmd in enumerate(self.commands):

            # Controls high throughput.
            # Basically just runs the same functions but forces certain defaults.
            # '-bulk example.csv' will run analysis for all smile strings in the example.csv file, otherwise it'll use the pdbs it finds
            # with the correct name.
            if cmd == '-bulk':

                csv_file = self.commands[count + 1]
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
                    start_point = bulk_data[name]['start'] if bulk_data[name]['start'] else 'rdkit_optimise'
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
                    if start_point == 'rdkit_optimise':

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
                        self.file = [file for file in files if file.endswith('.pdb') and not file.endswith('optimised.pdb')][0]

                        self.continue_log()

                    self.execute()
                    chdir('../')

                sys_exit('\nFinished bulk run. Use the command -progress to view which stages have completed.')

        # Check if an analysis is being done with restart / end arguments
        for count, cmd in enumerate(self.commands):

            if '-restart' in cmd or '-end' in cmd:
                if '-restart' in cmd:
                    start_point = self.commands[count + 1]
                    end_point = self.commands[count + 2]
                else:
                    start_point = 'rdkit_optimise'
                    end_point = self.commands[count + 1]

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

                if '-restart' in cmd:
                    # Set the file name based on the directory:
                    files = [file for file in listdir('.') if path.isfile(file)]
                    self.file = [file for file in files if file.endswith('.pdb') and not file.endswith('optimised.pdb')][0]

                    return self.file, self.commands

        # Finally, check if a single analysis is being done and if so, is it using a pdb or smiles.
        for count, cmd in enumerate(self.commands):

            if '-sm' in cmd:
                # Generate pdb from smiles string.
                self.file = smiles_to_pdb(self.commands[count + 1])
                self.defaults_dict['smiles string'] = self.commands[count + 1]

            # If a pdb is given instead, use that.
            elif 'pdb' in cmd:
                self.file = cmd

            print(self.start_up_msg)
            return self.file, self.commands

        else:
            sys_exit('\nYou did not ask QUBEKit to perform any kind of analysis, so it has stopped.\n'
                     'See the documentation (README) for details of acceptable commands with examples.\n'
                     'Try QUBEKit -sm C for an analysis of methane based on its smiles string.\n'
                     'Or, if you have not set up any configs yet, try QUBEKit -setup')

    def continue_log(self):
        """
        In the event of restarting an analysis, find and append to the existing log file
        rather than creating a new one.
        """

        with open(self.log_file, 'a+') as log_file:

            log_file.write(f'\n\nContinuing log file from previous execution: {datetime.now()}\n\n')
            log_file.write(f'The commands given were: {self.commands}\n\n')

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
            log_file.write(f'The commands given were: {self.commands}\n\n')
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

        if start_key in [key for key in self.order]:

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

                mol.pickle(state=self.order[next_key].__name__)

                return

    @staticmethod
    def rdkit_optimise(molecule):
        """Optimise the molecule coordinates using rdkit. Purely used to speed up bonds engine convergence."""

        copy(f'../{molecule.filename}', f'{molecule.filename}')
        molecule.filename, molecule.descriptors = smiles_mm_optimise(molecule.filename)

        # Initialise the molecule's pdb with its optimised form.
        molecule.read_pdb(input_type='mm')

        copy(f'{molecule.filename}', f'../{molecule.filename}')

        return molecule

    def parametrise(self, molecule):
        """Perform initial molecule parametrisation using OpenFF, Antechamber or XML."""

        copy(f'../{molecule.filename}', f'{molecule.filename}')
        # Parametrisation options:
        param_dict = {'openff': OpenFF, 'antechamber': AnteChamber, 'xml': XML}
        param_dict[self.fitting['parameter_engine']](molecule)

        append_to_log(f'Parametrised molecule with {self.fitting["parameter_engine"]}')

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

        append_to_log('Gaussian analysis complete' if self.qm['density_engine'] == 'g09' else 'ONETEP file made')

        return molecule

    def charges(self, molecule):
        """Perform DDEC calculation with Chargemol."""

        copy(f'../density/{molecule.name}.wfx', f'{molecule.name}.wfx')
        # TODO skip if onetep
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

    def torsion_optimise(self, molecule):
        """Perform torsion optimisation"""

        qm_engine = self.engine_dict[self.qm['bonds_engine']](molecule, self.all_configs)
        opt = TorsionOptimiser(molecule, qm_engine, self.all_configs, opt_method='BFGS', opls=True,
                               refinement_method='SP', vn_bounds=20)
        opt.run()

        append_to_log('Torsion optimisations complete')

        return molecule

    @staticmethod
    def finalise(molecule):
        """Make the xml and print the ligand object to terminal (in abbreviated form) and to the log file."""

        molecule.write_parameters()

        # Print ligand objects to log file and terminal
        pretty_print(molecule, to_file=True)
        pretty_print(molecule)

        return molecule

    @exception_logger_decorator
    def execute(self):
        """
        Calls all the relevant classes and methods for the full QM calculation in the correct order.
        Exceptions are added to log (if raised).
        """

        # Check if starting from the beginning; if so:
        if 'rdkit_optimise' in [key for key in self.order]:
            # Initialise ligand object fully before pickling it
            molecule = Ligand(self.file)
            molecule.pickle(state='rdkit_optimise')

        # Perform each key stage sequentially adding short messages (if given) to terminal to show progress.
        # Longer messages should be written inside the key stages' functions using helpers.append_to_log().
        # See PSI4 class in engines for an example of where this is used.
        self.stage_wrapper('rdkit_optimise', 'Partially optimising with rdkit', 'Optimisation complete')
        self.stage_wrapper('parametrise', 'Parametrising molecule', 'Molecule parametrised')
        self.stage_wrapper('qm_optimise', 'Optimising molecule, view .xyz file for progress', 'Molecule optimised')
        self.stage_wrapper('hessian', f'Calculating Hessian matrix')
        self.stage_wrapper('mod_sem', 'Calculating bonds and angles with modified Seminario method', 'Bonds and angles calculated')
        self.stage_wrapper('density', 'Performing density calculation with Gaussian09', 'Density calculation complete')
        self.stage_wrapper('charges', f'Chargemol calculating charges using DDEC{self.qm["ddec_version"]}', 'Charges calculated')
        self.stage_wrapper('lennard_jones', 'Performing Lennard-Jones calculation', 'Lennard-Jones parameters calculated')
        self.stage_wrapper('torsion_scan', 'Performing QM constrained optimisation with torsiondrive', 'Torsiondrive finished QM results saved')
        self.stage_wrapper('torsion_optimise', 'Performing torsion optimisation', 'Torsion optimisation complete')

        # This step is always performed
        self.stage_wrapper('finalise', 'Finalising analysis', 'Molecule analysis complete!')


def main():
    """This just stops the __repr__ automatically being called when the class is called."""
    Main()
