#!/usr/bin/env python


from QUBEKit.decorators import timer_logger

from csv import DictReader, writer, QUOTE_MINIMAL
from os import walk, listdir, path
from collections import OrderedDict
from numpy import allclose


@timer_logger
def config_loader(config_name='default_config'):
    """Sets up the desired global parameters from the config_file input.
    Allows different config settings for different projects, simply change the input config_name.
    """

    from importlib import import_module

    config = import_module(f'configs.{config_name}')

    return config.qm, config.fitting, config.descriptions


@timer_logger
def get_mol_data_from_csv(csv_name='sample_input.csv'):
    """Scan the csv file to find the row with the desired molecule data.
    Returns a dictionary of dictionaries in the form:
    {'methane': {'charge': 0, 'multiplicity': 1, ...}, 'ethane': {'charge': 0, ...}, ...}
    """

    with open(f'configs/{csv_name}', 'r') as csv_file:

        mol_confs = DictReader(csv_file)

        rows = []
        for row in mol_confs:

            # Converts to ordinary dict rather than ordered.
            row = dict(row)
            row['charge'] = int(float(row['charge']))
            row['multiplicity'] = int(float(row['multiplicity']))
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
        file_writer.writerow(['default', 0, 1, 'default_config', '', ''])

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


def pretty_print(mol, to_file=False, finished=True):
    """Takes a ligand molecule class object and displays all the class variables in a clean, readable format."""

    # Print to log: * On exception
    #               * On completion
    # Print to terminal: * On call
    #                    * On completion

    pre_string = f'\nOn {"completion" if finished else "exception"}, the ligand objects are:\n'

    # Print to log file
    if to_file:

        # Find log file name
        files = [file for file in listdir('.') if path.isfile(file)]
        qube_log_file = [file for file in files if file.startswith('QUBEKit_log')][0]

        with open(qube_log_file, 'a+') as log_file:

            log_file.write(pre_string)
            log_file.write(f'{mol.__str__()}')

    # Print to terminal
    else:

        print(pre_string)
        print(mol.__str__(trunc=True))


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

    # Check the matrix transpose is equal to the matrix within error.
    if not allclose(matrix, matrix.T, atol=error):
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
