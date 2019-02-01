#!/usr/bin/env python


from csv import DictReader, writer, QUOTE_MINIMAL
from QUBEKit.decorators import timer_logger, timer_func, for_all_methods
from os import walk
from collections import OrderedDict
from pathlib import Path
from os import path, system
from configparser import ConfigParser


class Configure:
    """Class to help load, read and write ini style configuration files returns dictionaries of the config
     settings as strings, all numbers must then be cast before use."""

    home = Path.home()
    config_folder = f'{home}/QUBEKit_configs/'
    master_file = 'master_config.ini'

    # QuBeKit config file allows users to reset the global variables

    qm = {
        'theory': 'B3LYP',  # Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example
        'basis': '6-311++G(d,p)',  # Basis set
        'vib_scaling': '0.991',  # Associated scaling to the theory
        'threads': '6',  # Number of processors used in g09; affects the bonds and dihedral scans
        'memory': '2',  # Amount of memory (in GB); specified in the g09 scripts
        'convergence': 'GAU_TIGHT',  # Criterion used during optimisations; works using psi4 and geometric so far
        'iterations': '100',  # Max number of optimisation iterations
        'bonds_engine': 'psi4',  # Engine used for bonds calculations
        'charges_engine': 'chargemol',  # Engine used for charges calculations
        'ddec_version': '6',  # DDEC version used by chargemol, 6 recommended but 3 is also available
        'geometric': 'True',  # Use geometric for optimised structure (if False, will just use psi4)
        'solvent': 'True',  # Use a solvent in the psi4/gaussian09 input
    }

    fitting = {
        'dih_start': '0',  # Starting angle of dihedral scan
        'increment': '15',  # Angle increase increment
        'num_scan': '25',  # Number of optimisations around the dihedral angle
        't_weight': 'infinity',  # Weighting temperature that can be changed to better fit complicated surfaces
        'l_pen': '0',  # The regularization penalty
        'new_dih_num': '501',  # Parameter number for the new dihedral to be fit
        'q_file': 'results.dat',  # If the results are collected with QuBeKit this is always true
        'tor_limit': '20',  # Torsion Vn limit to speed up fitting
        'div_index': '0',  # Fitting starting index in the division array
        'parameter_engine': 'openff',  # Method used for initial parametrisation
    }

    descriptions = {
        'chargemol': '/home/QUBEKit_user/chargemol_09_26_2017',  # Location of the chargemol program directory
        'log': '999',  # Default string for the working directories and logs
    }

    help = {
        'theory': ';Theory to use in freq and dihedral scans recommended wB97XD or B3LYP, for example',
        'basis': ';Basis set',
        'vib_scaling': ';Associated scaling to the theory',
        'threads': ';Number of processors used in g09; affects the bonds and dihedral scans',
        'memory': ';Amount of memory (in GB); specified in the g09 scripts',
        'convergence': ';Criterion used during optimisations; works using psi4 and geometric so far',
        'iterations': ';Max number of optimisation iterations',
        'bonds_engine': ';Engine used for bonds calculations',
        'charges_engine': ';Engine used for charges calculations',
        'ddec_version': ';DDEC version used by chargemol, 6 recommended but 3 is also available',
        'geometric': ';Use geometric for optimised structure (if False, will just use psi4)',
        'solvent': ';Use a solvent in the psi4/gaussian09 input',
        'dih_start': ';Starting angle of dihedral scan',
        'increment': ';Angle increase increment',
        'num_scan': ';Number of optimisations around the dihedral angle',
        't_weight': ';Weighting temperature that can be changed to better fit complicated surfaces',
        'l_pen': ';The regularization penalty',
        'new_dih_num': ';Parameter number for the new dihedral to be fit',
        'q_file': ';If the results are collected with QuBeKit this is always true',
        'tor_limit': ';Torsion Vn limit to speed up fitting',
        'div_index': ';Fitting starting index in the division array',
        'parameter_engine': ';Method used for initial parametrisation',
        'chargemol': ';Location of the chargemol program directory',
        'log': ';Default string for the working directories and logs'
    }

    @staticmethod
    def load_config(config_file='default_config'):
        """This method loads and returns the selected config file."""

        # Check if the default has been given
        if config_file == 'default_config':

            # Now check if the user has made a new master file that we should use
            if not Configure.check_master():
                # if there is no master then assign the default config
                qm, fitting, descriptions = Configure.qm, Configure.fitting, Configure.descriptions

            # else load the master file
            else:
                qm, fitting, descriptions = Configure.ini_parser(f'{Configure.config_folder+Configure.master_file}')

        else:
            # Load in the ini file given
            if path.exists(config_file):
                qm, fitting, descriptions = Configure.ini_parser(config_file)

            else:
                qm, fitting, descriptions = Configure.ini_parser(Configure.config_folder+config_file)

        # Now cast the numbers
        clean_ints = ['threads', 'memory', 'iterations', 'ddec_version', 'dih_start', 'increment',
                        'num_scan', 'new_dih_num', 'tor_limit', 'div_index']

        for key in clean_ints:

            if key in qm.keys():
                qm[key] = int(qm[key])

            elif key in fitting.keys():
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

        # Now handle the wight temp
        if fitting['t_weight'] != 'infinity':
            fitting['t_weight'] = float(fitting['t_weight'])

        # Now cast the regularization penalty to float
        fitting['l_pen'] = float(fitting['l_pen'])

        return qm, fitting, descriptions

    @staticmethod
    def ini_parser(ini):
        """parse an ini type config file and return the arguments as dictionaries."""

        config = ConfigParser(allow_no_value=True)
        config.read(ini)
        qm = config.__dict__['_sections']['QM']
        fitting = config.__dict__['_sections']['FITTING']
        descriptions = config.__dict__['_sections']['DESCRIPTIONS']

        return qm, fitting, descriptions

    @staticmethod
    def check_master():
        """Check if there is a new master ini file in the configs folder."""

        if path.exists(Configure.config_folder+Configure.master_file):
            return True
        else:
            return False

    @staticmethod
    def ini_writer(ini):
        """Make a new configuration file in the config folder using the current master as a template."""

        # make sure the ini file has an ini endding
        if not ini.endswith('.ini'):
            ini+='.ini'

        # Check the current master template
        if Configure.check_master():
            # if master then load
            qm, fitting, descriptions = Configure.ini_parser(Configure.config_folder+Configure.master_file)

        else:
            # If default is the config file then assign the defaults
            qm, fitting, descriptions = Configure.qm, Configure.fitting, Configure.descriptions

        # Set config parser to allow for comments
        config = ConfigParser(allow_no_value=True)
        config.add_section('QM')

        for key in qm.keys():
            config.set('QM', Configure.help[key])
            config.set('QM', key, qm[key])

        config.add_section('FITTING')

        for key in fitting.keys():
            config.set('FITTING', Configure.help[key])
            config.set('FITTING', key, fitting[key])

        config.add_section('DESCRIPTIONS')

        for key in descriptions.keys():
            config.set('DESCRIPTIONS', Configure.help[key])
            config.set('DESCRIPTIONS', key, descriptions[key])

        with open(f'{Configure.config_folder+ini}', 'w+')as out:
            config.write(out)

        return

    @staticmethod
    def get_name():
        """Ask the user for the name of the ini file"""

        name = input('Enter the name of the config file\n>')

        return name

    @staticmethod
    def ini_edit(ini):
        """Open the ini file for editing in the command line using whatever programme the user wants."""

        # make sure the ini file has an ini endding
        if not ini.endswith('.ini'):
            ini += '.ini'

        system(f'emacs -nw {Configure.config_folder+ini}')

        return


#TODO remove?
@timer_logger
def config_loader(config_name='default_config'):
    """Sets up the desired global parameters from the config_file input.
    Allows different config settings for different projects, simply change the input config_name.
    """

    # if config not supplied we need to check
    from importlib import import_module

    config = import_module(f'configs.{config_name}')

    return config.qm, config.fitting, config.descriptions


@timer_logger
def get_mol_data_from_csv(csv_name):
    """Scan the csv file to find the row with the desired molecule data.
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
            rows.append(row)

        # Creates the nested dictionaries with the names as the keys
        final = {rows[i]['name']: rows[i] for i in range(len(rows))}

        # Removes the names from the sub-dictionaries:
        # e.g. {'methane': {'name': 'methane', 'charge': 0, ...}, ...}
        # ---> {'methane': {'charge': 0, ...}, ...}
        for conf in final.keys():

            del final[conf]['name']

        return final


def generate_config_csv(csv_name):
    """Generates a csv with name "csv_name" with minimal information inside.
    Contains only headers and a row of defaults.
    """

    if csv_name[-4:] != '.csv':
        raise TypeError('Invalid or unspecified file type. File must be .csv')

    with open(csv_name, 'w') as csv_file:

        file_writer = writer(csv_file, delimiter=',', quotechar='|', quoting=QUOTE_MINIMAL)
        file_writer.writerow(['name', 'charge', 'multiplicity', 'config', 'smiles string', 'torsion order'])
        # file_writer.writerow(['default', 0, 1, 'default_config', '', ''])

    print(f'{csv_name} generated.')
    return


def append_to_log(log_file, message):
    """Appends a message to the log file in a specific format.
    Used for significant stages in the program such as when G09 has finished.
    """

    with open(log_file, 'a+') as file:
        file.write(f'~~~~~~~~{message.upper()}~~~~~~~~\n\n-------------------------------------------------------\n\n')


def get_overage(molecule):
    """Bodge."""

    overage_dict = {'methane': 12.0, 'ethane': 16.0, 'acetone': 20.0, 'benzene': 24.0, 'methanol': 17.0}
    return overage_dict[molecule]


def pretty_progress():
    """Neatly displays the state of all QUBEKit running directories in the terminal.
    Uses the log files to automatically generate a matrix which is then printed to screen in full colour 4k.
    """

    # Find the path of all files starting with QUBEKit_log and add their full path to log_files list
    log_files = []
    for root, dirs, files in walk('.', topdown=True):
        if 'QUBEKit' in root and '.tmp' not in root:
            log_files.append(f'{root}/QUBEKit_log_{root[10:]}')

    # Open all log files sequentially
    info = OrderedDict()
    for file in log_files:
        # Name is in the format 'moleculename_logname'
        name = file.split('_')[1] + '_' + file.split('_')[-1]

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
    for key_a, var_a in info.items():
        print('{:15}'.format(key_a[:13]), end=' ')
        # Inner dict contains the individual molecules' data.
        for key_b, var_b in info[key_a].items():
            if info[key_a][key_b] == 1:
                # Uses exit codes to set terminal font colours.
                # \033[ is the exit code. 1;32m are the style (bold); colour (green) m reenters the code block.
                # The second exit code resets the style back to default.
                print('\033[1;32m{:>12d}\033[0;0m'.format(info[key_a][key_b]), end=' ')
            else:
                print('\033[1;31m{:>12d}\033[0;0m'.format(info[key_a][key_b]), end=' ')
        # TODO Print tick mark after final column (use unicode characters).
        # TODO Change red to error, orange to not done yet.
        # TODO May need to improve formatting for longer molecule names.
        # Add blank line.
        print('')

    return


def pretty_print(mol, to_file=False):
    """Takes a ligand molecule class object and displays all the class variables in a clean, readable format."""

    # Print to log: * On exception
    #               * On completion
    # Print to terminal: * On call



    pass


def set_dict_val(file_name, search_term):
    """With a file open, search for a keyword; if it's anywhere in the file, return 1, else return 0."""

    with open(file_name, 'r+') as file:
        for line in file:
            if search_term in line:
                return 1
    return 0


def unpickle(pickle_jar):
    """Function to unpickle a set of ligand objects from the pickle file, and return a dictionary of ligands
    indexed by their progress.
    """

    from pickle import load

    mol_states = {}
    mols = []
    # unpickle the pickle jar
    # try to load a pickle file make sure to get all objects
    with open(pickle_jar, 'rb') as jar:
        while True:
            try:
                mols.append(load(jar))
            except:
                break
    # for each object in the jar put them into a dictionary indexed by there state
    for mol in mols:
        mol_states[mol.state] = mol

    return mol_states


def check_symmetry(matrix, error=0.00001):
    """Check matrix is symmetric to within some error."""

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if abs(matrix[i][j] - matrix[j][i]) > error:
                raise ValueError('Hessian is not symmetric.')

    print(f'Symmetry check successful. The matrix is symmetric within an error of {error}.')
    return True


def check_net_charge(charges, ideal_net=0, error=0.00001):
    """Given a list of charges, check if the calculated net charge is within error of the desired net charge."""

    # Ensure total charge is near to integer value:
    total_charge = sum(atom for atom in charges)

    if abs(total_charge - ideal_net) > error:
        raise ValueError('Total charge is not close enough to integer value.')

    print(f'Charge check successful. Net charge is within {error} of the desired net charge of {ideal_net}.')
    return True
