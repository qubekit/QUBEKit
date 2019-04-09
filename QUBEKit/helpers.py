#!/usr/bin/env python

from csv import DictReader, writer, QUOTE_MINIMAL
from os import walk, listdir, path, system
from collections import OrderedDict
from numpy import allclose
from pathlib import Path
from configparser import ConfigParser
from pickle import load
from contextlib import contextmanager


class Configure:
    """
    Class to help load, read and write ini style configuration files returns dictionaries of the config
    settings as strings, all numbers must then be cast before use.
    """

    home = Path.home()
    config_folder = f'{home}/QUBEKit_configs/'
    master_file = 'master_config.ini'

    # QuBeKit config file allows users to reset the global variables

    qm = {
        'theory': 'B3LYP',              # Theory to use in freq and dihedral scans recommended e.g. wB97XD or B3LYP
        'basis': '6-311++G(d,p)',       # Basis set
        'vib_scaling': '0.991',         # Associated scaling to the theory
        'threads': '6',                 # Number of processors used in Gaussian09; affects the bonds and dihedral scans
        'memory': '2',                  # Amount of memory (in GB); specified in the Gaussian09 scripts
        'convergence': 'GAU_TIGHT',     # Criterion used during optimisations; works using PSI4, GeomeTRIC and G09
        'iterations': '100',            # Max number of optimisation iterations
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
        'opt_method': 'BFGS',           # The type of scipy optimiser to use
        'refinement_method': 'SP',      # The type of QUBE refinement that should be done SP: single point energies
        'tor_limit': '20',              # Torsion Vn limit to speed up fitting
        'div_index': '0',               # Fitting starting index in the division array
        'parameter_engine': 'openff',   # Method used for initial parametrisation
        'l_pen': '0.0',                 # The regularisation penalty
    }

    descriptions = {
        'chargemol': '/home/QUBEKit_user/chargemol_09_26_2017',  # Location of the chargemol program directory
        'log': '999',                   # Default string for the working directories and logs
    }

    help = {
        'theory': ';Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example',
        'basis': ';Basis set',
        'vib_scaling': ';Associated scaling to the theory',
        'threads': ';Number of processors used in g09; affects the bonds and dihedral scans',
        'memory': ';Amount of memory (in GB); specified in the g09 and PSI4 scripts',
        'convergence': ';Criterion used during optimisations; works using psi4 and geometric so far',
        'iterations': ';Max number of optimisation iterations',
        'bonds_engine': ';Engine used for bonds calculations',
        'density_engine': ';Engine used to calculate the electron density',
        'charges_engine': ';Engine used for charge partitioning',
        'ddec_version': ';DDEC version used by chargemol, 6 recommended but 3 is also available',
        'geometric': ';Use geometric for optimised structure (if False, will just use PSI4)',
        'solvent': ';Use a solvent in the psi4/gaussian09 input',
        'dih_start': ';Starting angle of dihedral scan',
        'increment': ';Angle increase increment',
        'dih_end': ';The last dihedral angle in the scan',
        't_weight': ';Weighting temperature that can be changed to better fit complicated surfaces',
        'l_pen': ';The regularisation penalty',
        'opt_method': ';The type of scipy optimiser to use',
        'refinement_method': ';The type of QUBE refinement that should be done SP: single point energies',
        'tor_limit': ';Torsion Vn limit to speed up fitting',
        'div_index': ';Fitting starting index in the division array',
        'parameter_engine': ';Method used for initial parametrisation',
        'chargemol': ';Location of the chargemol program directory (do not end with a "/")',
        'log': ';Default string for the working directories and logs'
    }

    @staticmethod
    def load_config(config_file='default_config'):
        """This method loads and returns the selected config file."""

        if config_file == 'default_config':

            # Check if the user has made a new master file to use
            if Configure.check_master():
                qm, fitting, descriptions = Configure.ini_parser(f'{Configure.config_folder + Configure.master_file}')

            else:
                # If there is no master then assign the default config
                qm, fitting, descriptions = Configure.qm, Configure.fitting, Configure.descriptions

        else:
            # Load in the ini file given
            if path.exists(config_file):
                qm, fitting, descriptions = Configure.ini_parser(config_file)

            else:
                qm, fitting, descriptions = Configure.ini_parser(Configure.config_folder + config_file)

        # Now cast the numbers
        clean_ints = ['threads', 'memory', 'iterations', 'ddec_version', 'dih_start',
                      'increment', 'dih_end', 'tor_limit', 'div_index']

        for key in clean_ints:

            if key in qm:
                qm[key] = int(qm[key])

            elif key in fitting:
                fitting[key] = int(fitting[key])

        # Now cast the one float the scaling
        qm['vib_scaling'] = float(qm['vib_scaling'])

        # Now cast the bools
        if qm['geometric'].lower() == 'true':
            qm['geometric'] = True

        else:
            qm['geometric'] = False

        if qm['solvent'].lower() == 'true':
            qm['solvent'] = True

        else:
            qm['solvent'] = False

        # Now handle the weight temp
        if fitting['t_weight'] != 'infinity':
            fitting['t_weight'] = float(fitting['t_weight'])

        # Now cast the regularisation penalty to float
        fitting['l_pen'] = float(fitting['l_pen'])

        return qm, fitting, descriptions

    @staticmethod
    def ini_parser(ini):
        """Parse an ini type config file and return the arguments as dictionaries."""

        config = ConfigParser(allow_no_value=True)
        config.read(ini)
        qm = config.__dict__['_sections']['QM']
        fitting = config.__dict__['_sections']['FITTING']
        descriptions = config.__dict__['_sections']['DESCRIPTIONS']

        return qm, fitting, descriptions

    @staticmethod
    def show_ini():
        """Show all of the ini file options in the config folder."""

        inis = listdir(Configure.config_folder)

        return inis

    @staticmethod
    def check_master():
        """Check if there is a new master ini file in the configs folder."""

        return True if path.exists(Configure.config_folder + Configure.master_file) else False

    @staticmethod
    def ini_writer(ini):
        """Make a new configuration file in the config folder using the current master as a template."""

        # make sure the ini file has an ini ending
        if not ini.endswith('.ini'):
            ini += '.ini'

        # Check the current master template
        if Configure.check_master():
            # If master then load
            qm, fitting, descriptions = Configure.ini_parser(Configure.config_folder + Configure.master_file)

        else:
            # If default is the config file then assign the defaults
            qm, fitting, descriptions = Configure.qm, Configure.fitting, Configure.descriptions

        # Set config parser to allow for comments
        config = ConfigParser(allow_no_value=True)
        config.add_section('QM')

        for key, val in qm.items():
            config.set('QM', Configure.help[key])
            config.set('QM', key, val)

        config.add_section('FITTING')

        for key, val in fitting.items():
            config.set('FITTING', Configure.help[key])
            config.set('FITTING', key, val)

        config.add_section('DESCRIPTIONS')

        for key, val in descriptions.items():
            config.set('DESCRIPTIONS', Configure.help[key])
            config.set('DESCRIPTIONS', key, val)

        with open(f'{Configure.config_folder + ini}', 'w+') as out:
            config.write(out)

    @staticmethod
    def ini_edit(ini_file):
        """Open the ini file for editing in the command line using whatever program the user wants."""

        # Make sure the ini file has an ini ending
        if not ini_file.endswith('.ini'):
            ini_file += '.ini'

        system(f'emacs -nw {Configure.config_folder + ini_file}')

        return


def mol_data_from_csv(csv_name):
    """
    Scan the csv file to find the row with the desired molecule data.
    Returns a dictionary of dictionaries in the form:
    {'methane': {'charge': 0, 'multiplicity': 1, ...}, 'ethane': {'charge': 0, ...}, ...}
    """

    with open(csv_name, 'r') as csv_file:

        mol_confs = DictReader(csv_file)

        rows = []
        for row in mol_confs:

            # Converts to ordinary dict rather than ordered.
            row = dict(row)
            row['charge'] = int(float(row['charge']))
            row['multiplicity'] = int(float(row['multiplicity']))
            # If there is no config given assume its the default
            row['config'] = row['config'] if row['config'] else 'default_config'
            # Converts empty string to None (looks a bit weird, I know) otherwise leaves it alone.
            row['smiles string'] = row['smiles string'] if row['smiles string'] else None
            row['torsion order'] = row['torsion order'] if row['torsion order'] else None
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


def generate_bulk_csv(csv_name):
    """
    Generates a csv with name "csv_name" with minimal information inside.
    Contains only headers and a row of defaults.
    """

    if csv_name[-4:] != '.csv':
        raise TypeError('Invalid or unspecified file type. File must be .csv')

    with open(csv_name, 'w') as csv_file:

        file_writer = writer(csv_file, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        file_writer.writerow(['name', 'charge', 'multiplicity', 'config', 'smiles string', 'torsion order', 'start', 'end'])

    print(f'{csv_name} generated.')
    return


def append_to_log(message, msg_type='major'):
    """
    Appends a message to the log file in a specific format.
    Used for significant stages in the program such as when G09 has finished.
    """

    # Check if the message is a blank string to avoid adding blank lines and separators
    if message:
        with open('../QUBEKit_log.txt', 'a+') as file:
            if msg_type == 'major':
                file.write(f'~~~~~~~~{message.upper()}~~~~~~~~')
            elif msg_type == 'warning':
                file.write(f'########{message.upper()}########')
            elif msg_type == 'minor':
                file.write(f'~~~~~~~~{message}~~~~~~~~')

            file.write(f'\n\n{"-" * 50}\n\n')


def get_overage(molecule):
    """Bodge."""

    overage_dict = {'methane': 12.0, 'ethane': 16.0, 'acetone': 20.0, 'benzene': 24.0, 'methanol': 17.0}
    return overage_dict[molecule]


def pretty_progress():
    """
    Neatly displays the state of all QUBEKit running directories in the terminal.
    Uses the log files to automatically generate a matrix which is then printed to screen in full colour 4k.
    """

    # TODO Add legend.
    # TODO Print tick mark after final column (use unicode characters).
    # TODO May need to improve formatting for longer molecule names.

    # Find the path of all files starting with QUBEKit_log and add their full path to log_files list
    log_files = []
    for root, dirs, files in walk('.', topdown=True):
        for file in files:
            if 'QUBEKit_log.txt' in file:
                log_files.append(path.abspath(f'{root}/{file}'))

    # Open all log files sequentially
    info = OrderedDict()
    for file in log_files:
        with open(file, 'r') as log_file:
            for line in log_file:
                if 'Analysing:' in line:
                    name = line.split()[1]
                    break
            else:
                raise EOFError('Cannot locate molecule name in file.')

        # Create ordered dictionary based on the log file info
        info[name] = OrderedDict()

        # Set the values of the ordered dicts based on the info in the log files.
        # Tildes (~) are used as markers for useful information.
        info[name]['parametrised'] = set_dict_val(file, '~PARAMETRISED')
        info[name]['optimised'] = set_dict_val(file, '~OPTIMISED')
        info[name]['mod sem'] = set_dict_val(file, '~MODIFIED')
        info[name]['gaussian'] = set_dict_val(file, '~GAUSSIAN')
        info[name]['chargemol'] = set_dict_val(file, '~CHARGEMOL')
        info[name]['lennard'] = set_dict_val(file, '~LENNARD')
        info[name]['torsions'] = set_dict_val(file, '~TORSION')

    header_string = '{:15} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}'
    print(header_string.format('Name', 'Parametrised', 'Optimised', 'Mod-Sem', 'Gaussian', 'Chargemol', 'L-J', 'Torsions'))

    # Outer dict contains the names of the molecules.
    for key_out, var_out in info.items():
        print(f'{key_out[:13]:15}', end=' ')

        # Inner dict contains the individual molecules' data.
        for var_in in var_out.values():
            if var_in == 1:
                # Uses exit codes to set terminal font colours.
                # \033[ is the exit code. 1;32m are the style (bold); colour (green) m reenters the code block.
                # The second exit code resets the style back to default.
                print(f'\033[1;32m{var_in:>12d}\033[0;0m', end=' ')

            else:
                print(f'\033[1;31m{var_in:>12d}\033[0;0m', end=' ')

        print('')


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


def set_dict_val(file_name, search_term):
    """With a file open, search for a keyword; if it's anywhere in the file, return 1, else return 0."""

    with open(file_name, 'r+') as file:
        for line in file:
            if search_term in line:
                return True
    return False


def unpickle():
    """
    Function to unpickle a set of ligand objects from the pickle file, and return a dictionary of ligands
    indexed by their progress.
    """

    mol_states = OrderedDict()

    # unpickle the pickle jar
    # try to load a pickle file make sure to get all objects
    with open('.QUBEKit_states', 'rb') as jar:
        while True:
            try:
                mol = load(jar)
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
        assert (allclose(matrix, matrix.T, atol=error)), 'Matrix is not symmetric.'

    print(f'Symmetry check successful. The matrix is symmetric within an error of {error}.')
    return True


def check_net_charge(charges, ideal_net=0, error=0.00001):
    """Given a list of charges, check if the calculated net charge is within error of the desired net charge."""

    # Ensure total charge is near to integer value:
    total_charge = sum(atom for atom in charges)

    with assert_wrapper(ValueError):
        assert (abs(total_charge - ideal_net) < error), 'Total charge is not close enough to desired integer value in configs.'

    print(f'Charge check successful. Net charge is within {error} of the desired net charge of {ideal_net}.')
    return True
