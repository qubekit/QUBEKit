#!/usr/bin/env python3

from collections import OrderedDict
from configparser import ConfigParser
from contextlib import contextmanager
import csv
from functools import partial
import math
import os
from pathlib import Path
import pickle

import numpy as np


class Configure:
    """
    Class to help load, read and write ini style configuration files returns dictionaries of the config
    settings as strings, all numbers must then be cast before use.
    """

    # TODO Use proper pathing (os.path.join or similar)

    home = Path.home()
    config_folder = f'{home}/QUBEKit_configs/'
    master_file = 'master_config.ini'

    qm = {
        'theory': 'B3LYP',              # Theory to use in freq and dihedral scans recommended e.g. wB97XD or B3LYP
        'basis': '6-311++G(d,p)',       # Basis set
        'vib_scaling': '0.967',         # Associated scaling to the theory
        'threads': '2',                 # Number of processors used in Gaussian09; affects the bonds and dihedral scans
        'memory': '2',                  # Amount of memory (in GB); specified in the Gaussian09 scripts
        'convergence': 'GAU_TIGHT',     # Criterion used during optimisations; works using PSI4, GeomeTRIC and G09
        'iterations': '350',            # Max number of optimisation iterations
        'bonds_engine': 'psi4',         # Engine used for bonds calculations
        'density_engine': 'g09',        # Engine used to calculate the electron density
        'charges_engine': 'chargemol',  # Engine used for charge partitioning
        'ddec_version': '6',            # DDEC version used by Chargemol, 6 recommended but 3 is also available
        'geometric': 'True',            # Use GeomeTRIC for optimised structure (if False, will just use PSI4)
        'solvent': 'True',              # Use a solvent in the PSI4/Gaussian09 input
    }

    fitting = {
        'dih_start': '-165',            # Starting angle of dihedral scan
        'increment': '15',              # Angle increase increment
        'dih_end': '180',               # The last dihedral angle in the scan
        't_weight': 'infinity',         # Weighting temperature that can be changed to better fit complicated surfaces
        'opt_method': 'BFGS',           # The type of SciPy optimiser to use
        'refinement_method': 'SP',      # The type of QUBE refinement that should be done SP: single point energies
        'tor_limit': '20',              # Torsion Vn limit to speed up fitting
        'div_index': '0',               # Fitting starting index in the division array
        'parameter_engine': 'xml',      # Method used for initial parametrisation
        'l_pen': '0.0',                 # The regularisation penalty
        'relative_to_global': 'False'   # If we should compute our relative energy surface
                                        # compared to the global minimum
    }

    excited = {
        'excited_state': 'False',       # Is this an excited state calculation
        'excited_theory': 'TDA',
        'nstates': '3',
        'excited_root': '1',
        'use_pseudo': 'False',
        'pseudo_potential_block': ''
    }

    descriptions = {
        'chargemol': '/home/<QUBEKit_user>/chargemol_09_26_2017',  # Location of the chargemol program directory
        'log': '999',                   # Default string for the working directories and logs
    }

    help = {
        'theory': ';Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example',
        'basis': ';Basis set',
        'vib_scaling': ';Associated scaling to the theory',
        'threads': ';Number of processors used in g09; affects the bonds and dihedral scans',
        'memory': ';Amount of memory (in GB); specified in the g09 and PSI4 scripts',
        'convergence': ';Criterion used during optimisations; GAU, GAU_TIGHT, GAU_VERYTIGHT',
        'iterations': ';Max number of optimisation iterations',
        'bonds_engine': ';Engine used for bonds calculations',
        'density_engine': ';Engine used to calculate the electron density',
        'charges_engine': ';Engine used for charge partitioning',
        'ddec_version': ';DDEC version used by Chargemol, 6 recommended but 3 is also available',
        'geometric': ';Use geometric for optimised structure (if False, will just use PSI4)',
        'solvent': ';Use a solvent in the psi4/gaussian09 input',
        'dih_start': ';Starting angle of dihedral scan',
        'increment': ';Angle increase increment',
        'dih_end': ';The last dihedral angle in the scan',
        't_weight': ';Weighting temperature that can be changed to better fit complicated surfaces',
        'l_pen': ';The regularisation penalty',
        'relative_to_global': ';If we should compute our relative energy surface compared to the global minimum',
        'opt_method': ';The type of SciPy optimiser to use',
        'refinement_method': ';The type of QUBE refinement that should be done SP: single point energies',
        'tor_limit': ';Torsion Vn limit to speed up fitting',
        'div_index': ';Fitting starting index in the division array',
        'parameter_engine': ';Method used for initial parametrisation',
        'chargemol': ';Location of the Chargemol program directory (do not end with a "/")',
        'log': ';Default string for the names of the working directories',
        'excited_state': ';Use the excited state',
        'excited_theory': ';Excited state theory TDA or TD',
        'nstates': ';The number of states to use',
        'excited_root': ';The root',
        'use_pseudo': ';Use a pseudo potential',
        'pseudo_potential_block': ';Enter the pseudo potential block here eg'
    }

    def load_config(self, config_file='default_config'):
        """This method loads and returns the selected config file."""

        if config_file == 'default_config':

            # Check if the user has made a new master file to use
            if self.check_master():
                qm, fitting, excited, descriptions = self.ini_parser(f'{self.config_folder + self.master_file}')

            else:
                # If there is no master then assign the default config
                qm, fitting, excited, descriptions = self.qm, self.fitting, self.excited, self.descriptions

        else:
            # Load in the ini file given
            if os.path.exists(config_file):
                qm, fitting, excited, descriptions = self.ini_parser(config_file)

            else:
                qm, fitting, excited, descriptions = self.ini_parser(self.config_folder + config_file)

        # Now cast the numbers
        clean_ints = ['threads', 'memory', 'iterations', 'ddec_version', 'dih_start',
                      'increment', 'dih_end', 'tor_limit', 'div_index', 'nstates', 'excited_root']

        for key in clean_ints:

            if key in qm:
                qm[key] = int(qm[key])

            elif key in fitting:
                fitting[key] = int(fitting[key])

            elif key in excited:
                excited[key] = int(excited[key])

        # Now cast the one float the scaling
        qm['vib_scaling'] = float(qm['vib_scaling'])

        # Now cast the bools
        qm['geometric'] = True if qm['geometric'].lower() == 'true' else False
        qm['solvent'] = True if qm['solvent'].lower() == 'true' else False
        excited['excited_state'] = True if excited['excited_state'].lower() == 'true' else False
        excited['use_pseudo'] = True if excited['use_pseudo'].lower() == 'true' else False
        fitting['relative_to_global'] = True if fitting['relative_to_global'].lower() == 'true' else False

        # Now handle the weight temp
        if fitting['t_weight'] != 'infinity':
            fitting['t_weight'] = float(fitting['t_weight'])

        # Now cast the regularisation penalty to float
        fitting['l_pen'] = float(fitting['l_pen'])

        # return qm, fitting, descriptions
        return {**qm, **fitting, **excited, **descriptions}

    @staticmethod
    def ini_parser(ini):
        """Parse an ini type config file and return the arguments as dictionaries."""

        config = ConfigParser(allow_no_value=True)
        config.read(ini)
        qm = config.__dict__['_sections']['QM']
        fitting = config.__dict__['_sections']['FITTING']
        excited = config.__dict__['_sections']['EXCITED']
        descriptions = config.__dict__['_sections']['DESCRIPTIONS']

        return qm, fitting, excited, descriptions

    def show_ini(self):
        """Show all of the ini file options in the config folder."""

        # Hide the emacs backups
        return [ini for ini in os.listdir(self.config_folder) if not ini.endswith('~')]

    def check_master(self):
        """Check if there is a new master ini file in the configs folder."""

        return True if os.path.exists(self.config_folder + self.master_file) else False

    def ini_writer(self, ini):
        """Make a new configuration file in the config folder using the current master as a template."""

        # make sure the ini file has an ini ending
        if not ini.endswith('.ini'):
            ini += '.ini'

        # Load a new configs from the options
        qm, fitting, excited, descriptions = self.qm, self.fitting, self.excited, self.descriptions

        # Set config parser to allow for comments
        config = ConfigParser(allow_no_value=True)
        config.add_section('QM')

        for key, val in qm.items():
            config.set('QM', self.help[key])
            config.set('QM', key, val)

        config.add_section('FITTING')

        for key, val in fitting.items():
            config.set('FITTING', self.help[key])
            config.set('FITTING', key, val)

        config.add_section('EXCITED')

        for key, val in excited.items():
            config.set('EXCITED', self.help[key])
            config.set('EXCITED', key, val)

        config.add_section('DESCRIPTIONS')

        for key, val in descriptions.items():
            config.set('DESCRIPTIONS', self.help[key])
            config.set('DESCRIPTIONS', key, val)

        with open(f'{self.config_folder + ini}', 'w+') as out:
            config.write(out)

    def ini_edit(self, ini_file):
        """Open the ini file for editing in the command line using whatever program the user wants."""

        # Make sure the ini file has an ini ending
        if not ini_file.endswith('.ini'):
            ini_file += '.ini'

        os.system(f'emacs -nw {self.config_folder + ini_file}')


def mol_data_from_csv(csv_name):
    """
    Scan the csv file to find the row with the desired molecule data.
    Returns a dictionary of dictionaries in the form:
    {'methane': {'charge': 0, 'multiplicity': 1, ...}, 'ethane': {'charge': 0, ...}, ...}
    """

    with open(csv_name, 'r') as csv_file:

        mol_confs = csv.DictReader(csv_file)

        rows = []
        for row in mol_confs:

            # Converts to ordinary dict rather than ordered.
            row = dict(row)
            # If there is no config given assume its the default
            row['charge'] = int(float(row['charge'])) if row['charge'] else 0
            row['multiplicity'] = int(float(row['multiplicity'])) if row['multiplicity'] else 1
            row['config'] = row['config'] if row['config'] else 'default_config'
            row['smiles'] = row['smiles'] if row['smiles'] else None
            row['torsion_order'] = row['torsion_order'] if row['torsion_order'] else None
            row['restart'] = row['restart'] if row['restart'] else None
            row['end'] = row['end'] if row['end'] else 'finalise'
            rows.append(row)

    # Creates the nested dictionaries with the names as the keys
    final = {row['name']: row for row in rows}

    # Removes the names from the sub-dictionaries:
    # e.g. {'methane': {'name': 'methane', 'charge': 0, ...}, ...}
    # ---> {'methane': {'charge': 0, ...}, ...}
    for val in final.values():
        del val['name']

    return final


def generate_bulk_csv(csv_name, max_execs=None):
    """
    Generates a csv with name "csv_name" with minimal information inside.
    Contains only headers and a row of defaults and populates all of the named files where available.
    max_execs determines the max number of executions per csv file.
    For example, 10 pdb files with a value of max_execs=6 will generate two csv files,
    one containing 6 of those files, the other with the remaining 4.
    """

    if csv_name[-4:] != '.csv':
        raise TypeError('Invalid or unspecified file type. File must be .csv')

    # Find any local pdb files to write sample configs
    files = []
    for file in os.listdir("."):
        if file.endswith('.pdb'):
            files.append(file[:-4])

    # If max number of pdbs per file is unspecified, just put them all in one file.
    if max_execs is None:
        with open(csv_name, 'w') as csv_file:

            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['name', 'charge', 'multiplicity', 'config', 'smiles', 'torsion_order', 'restart', 'end'])
            for file in files:
                file_writer.writerow([file, 0, 1, '', '', '', '', ''])
        print(f'{csv_name} generated.', flush=True)
        return

    try:
        max_execs = int(max_execs)
    except TypeError:
        raise TypeError('Number of executions must be provided as an int greater than 1.')
    if max_execs > len(files):
        raise ValueError('Number of executions cannot exceed the number of files provided.')

    # If max number of pdbs per file is specified, spread them across several csv files.
    num_csvs = math.ceil(len(files) / max_execs)

    for csv_count in range(num_csvs):
        with open(f'{csv_name[:-4]}_{str(csv_count).zfill(2)}.csv', 'w') as csv_file:
            file_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            file_writer.writerow(['name', 'charge', 'multiplicity', 'config', 'smiles', 'torsion_order', 'restart', 'end'])

            for file in files[csv_count * max_execs: (csv_count + 1) * max_execs]:
                file_writer.writerow([file, 0, 1, '', '', '', '', ''])

        print(f'{csv_name[:-4]}_{str(csv_count).zfill(2)}.csv generated.', flush=True)


def append_to_log(message, msg_type='major'):
    """
    Appends a message to the log file in a specific format.
    Used for significant stages in the program such as when a stage has finished.
    """

    # Starting in the current directory walk back looking for the log file
    # Stop at the first file found this should be our file
    search_dir = os.getcwd()

    while True:
        if 'QUBEKit_log.txt' in os.listdir(search_dir):
            log_file = os.path.abspath(os.path.join(search_dir, 'QUBEKit_log.txt'))
            break

        # Else we have to split the search path
        else:
            search_dir = os.path.split(search_dir)[0]
            if not search_dir:
                raise FileNotFoundError('Cannot locate QUBEKit log file.')

    # Check if the message is a blank string to avoid adding blank lines and unnecessary separators
    if message:
        with open(log_file, 'a+') as file:
            if msg_type == 'major':
                file.write(f'~~~~~~~~{message.upper()}~~~~~~~~')
            elif msg_type == 'warning':
                file.write(f'########{message.upper()}########')
            elif msg_type == 'minor':
                file.write(f'~~~~~~~~{message}~~~~~~~~')
            else:
                raise KeyError('Invalid message type; use major, warning or minor.')

            file.write(f'\n\n{"-" * 50}\n\n')


def pretty_progress():
    """
    Neatly displays the state of all QUBEKit running directories in the terminal.
    Uses the log files to automatically generate a matrix which is then printed to screen in full colour 4k.
    """

    printf = partial(print, flush=True)

    # Find the path of all files starting with QUBEKit_log and add their full path to log_files list
    log_files = []
    for root, dirs, files in os.walk('.', topdown=True):
        for file in files:
            if 'QUBEKit_log.txt' in file and 'backups' not in root:
                log_files.append(os.path.abspath(f'{root}/{file}'))

    if not log_files:
        print('No QUBEKit directories with log files found. Perhaps you need to move to the parent directory.')
        return

    # Open all log files sequentially
    info = OrderedDict()
    for file in log_files:
        with open(file, 'r') as log_file:
            for line in log_file:
                if 'Analysing:' in line:
                    name = line.split()[1]
                    break
            else:
                # If the molecule name isn't found, there's something wrong with the log file
                # To avoid errors, just skip over that file and tell the user.
                print(f'Cannot locate molecule name in {file}\nIs it a valid, QUBEKit-made log file?\n')

        # Create ordered dictionary based on the log file info
        info[name] = populate_progress_dict(file)

    # Uses exit codes to set terminal font colours.
    # \033[ is the exit code. 1;32m are the style (bold); colour (green) m reenters the code block.
    # The second exit code resets the style back to default.

    # Need to add an end tag or terminal colours will persist
    end = '\033[0m'

    # Bold colours
    colours = {
        'red': '\033[1;31m',
        'green': '\033[1;32m',
        'orange': '\033[1;33m',
        'blue': '\033[1;34m',
        'purple': '\033[1;35m'
    }

    printf('Displaying progress of all analyses in current directory.')
    printf(f'Progress key: {colours["green"]}\u2713{end} = Done;', end=' ')
    printf(f'{colours["blue"]}S{end} = Skipped;', end=' ')
    printf(f'{colours["red"]}E{end} = Error;', end=' ')
    printf(f'{colours["orange"]}R{end} = Running;', end=' ')
    printf(f'{colours["purple"]}~{end} = Queued')

    header_string = '{:15}' + '{:>10}' * 10
    printf(header_string.format(
        'Name', 'Param', 'MM Opt', 'QM Opt', 'Hessian', 'Mod-Sem', 'Density', 'Charges', 'L-J', 'Tor Scan', 'Tor Opt'))

    # Sort the info alphabetically
    info = OrderedDict(sorted(info.items(), key=lambda tup: tup[0]))

    # Outer dict contains the names of the molecules.
    for key_out, var_out in info.items():
        printf(f'{key_out[:13]:15}', end=' ')

        # Inner dict contains the individual molecules' data.
        for var_in in var_out.values():

            if var_in == u'\u2713':
                printf(f'{colours["green"]}{var_in:>9}{end}', end=' ')

            elif var_in == 'S':
                printf(f'{colours["blue"]}{var_in:>9}{end}', end=' ')

            elif var_in == 'E':
                printf(f'{colours["red"]}{var_in:>9}{end}', end=' ')

            elif var_in == 'R':
                printf(f'{colours["orange"]}{var_in:>9}{end}', end=' ')

            elif var_in == '~':
                printf(f'{colours["purple"]}{var_in:>9}{end}', end=' ')

        printf('')


def populate_progress_dict(file_name):
    """
    With a log file open:
        Search for a keyword marking the completion or skipping of a stage;
        If that's not found, look for error messages,
        Otherwise, just return that the stage hasn't finished yet.
    Key:
        tick mark: Done; S: Skipped; E: Error; ~ (tilde): Not done yet, no error found.
    """

    # Indicators in the log file which show a stage has completed
    search_terms = ['PARAMETRISATION', 'MM_OPT', 'QM_OPT', 'HESSIAN', 'MOD_SEM', 'DENSITY', 'CHARGE', 'LENNARD',
                    'TORSION_S', 'TORSION_O']

    progress = OrderedDict((term, '~') for term in search_terms)

    restart_log = False

    with open(file_name, 'r') as file:
        for line in file:

            # Reset progress when restarting (set all progress to incomplete)
            if 'Continuing log file' in line:
                restart_log = True

            # Look for the specific search terms
            for term in search_terms:
                if term in line:
                    # If you find a search term, check if it's skipped (S)
                    if 'SKIP' in line:
                        progress[term] = 'S'
                    # If we have restarted then we need to
                    elif 'STARTING' in line:
                        progress[term] = 'R'
                    # If its finishing tag is present it is done (tick)
                    elif 'FINISHING' in line:
                        progress[term] = u'\u2713'
                        last_success = term

            # If an error is found, then the stage after the last successful stage has errored (E)
            if 'Exception Logger - ERROR' in line:
                # On the rare occasion that the error occurs after torsion optimisation (the final stage),
                # a try except is needed to catch the index error (because there's no stage after torsion_optimisation).
                try:
                    term = search_terms[search_terms.index(last_success) + 1]
                except IndexError:
                    term = search_terms[search_terms.index(last_success)]
                # If errored immediately, then last_success won't have been defined yet
                except UnboundLocalError:
                    term = 'PARAMETRISATION'
                progress[term] = 'E'

    if restart_log:
        for term, stage in progress.items():
            # Find where the program was restarted from
            if stage == 'R':
                restart_term = search_terms.index(term)
                break
        else:
            # If no stage is running, find the first stage that hasn't started; the first `~`
            for term, stage in progress.items():
                if stage == '~':
                    restart_term = search_terms.index(term)
                    break
            else:
                raise UnboundLocalError(
                    'Cannot find where QUBEKit was restarted from. Please check the log file for progress.')

        # Reset anything after the restart term to be `~` even if it was previously completed.
        for term in search_terms[restart_term + 1:]:
            progress[term] = '~'

    return progress


def pretty_print(molecule, to_file=False, finished=True):
    """
    Takes a ligand molecule class object and displays all the class variables in a clean, readable format.

    Print to log: * On exception
                  * On completion
    Print to terminal: * On call
                       * On completion
    """

    pre_string = f'\n\nOn {"completion" if finished else "exception"}, the ligand objects are:'

    # Print to log file rather than to terminal
    if to_file:
        with open(f'../QUBEKit_log.txt', 'a+') as log_file:
            log_file.write(f'{pre_string.upper()}\n\n{molecule.__str__()}')

    # Print to terminal
    else:
        print(pre_string)
        # Custom __str__ method; see its documentation for details.
        print(molecule.__str__(trunc=True))
        print('')


def unpickle():
    """
    Function to unpickle a set of ligand objects from the pickle file, and return a dictionary of ligands
    indexed by their progress.
    """

    mol_states = OrderedDict()

    # unpickle the pickle jar
    # try to load a pickle file make sure to get all objects
    pickle_file = f'{"" if ".QUBEKit_states" in os.listdir(".") else "../"}.QUBEKit_states'
    with open(pickle_file, 'rb') as jar:
        while True:
            try:
                mol = pickle.load(jar)
                mol_states[mol.state] = mol
            except EOFError:
                break

    return mol_states


@contextmanager
def assert_wrapper(exception_type):
    """
    Makes assertions more informative when an Exception is thrown.
    Rather than just getting 'AssertionError' all the time, an actual named exception can be passed.
    Can be called multiple times in the same 'with' statement for the same exception type but different exceptions.

    Simple example use cases:

        with assert_wrapper(ValueError):
            assert (arg1 > 0), 'arg1 cannot be non-positive.'
            assert (arg2 != 12), 'arg2 cannot be 12.'
        with assert_wrapper(TypeError):
            assert (type(arg1) is not float), 'arg1 must not be a float.'
    """

    try:
        yield
    except AssertionError as excep:
        raise exception_type(*excep.args)


def check_symmetry(matrix, error=0.00001):
    """Check matrix is symmetric to within some error."""

    # Check the matrix transpose is equal to the matrix within error.
    with assert_wrapper(ValueError):
        assert (np.allclose(matrix, matrix.T, atol=error)), 'Matrix is not symmetric.'

    print(f'Symmetry check successful. The matrix is symmetric within an error of {error}.')
    return True


def check_net_charge(charges, ideal_net=0, error=0.00001):
    """Given a list of charges, check if the calculated net charge is within error of the desired net charge."""

    # Ensure total charge is near to integer value:
    total_charge = sum(atom for atom in charges)

    with assert_wrapper(ValueError):
        assert (abs(total_charge - ideal_net) < error), ('Total charge is not close enough to desired '
                                                         'integer value in configs.')

    print(f'Charge check successful. Net charge is within {error} of the desired net charge of {ideal_net}.')
    return True
