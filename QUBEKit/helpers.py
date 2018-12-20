#!/usr/bin/env python


from csv import DictReader, writer, QUOTE_MINIMAL
from QUBEKit.decorators import timer_logger
from os import walk
from collections import OrderedDict


def pdb_to_coord_list(pdb_file):
    """Opens a pdb file ('methane.pdb' for example) and returns the molecule name, followed by a list of the coords.
    Coords will be of the form: [['C', -1.28, 0.127, -0.003], ['C', 0.029, -0.428, -0.47] ... ]
    """

    # TODO Ensure this works with different pdb styles.

    molecule = []

    with open(pdb_file, 'r') as file:
        lines = file.readlines()
        # First line contains the molecule name as the last entry.
        molecule_name = lines[0].split()[1]

        for line in lines:
            if 'ATOM' in line or 'HETATM' in line:
                # Create a list containing the element's name and coords, e.g. ['C', -1.28, 0.127, -0.003]
                element_name = [line.split()[-1]]
                coords = line.split()[-6:-3]

                atom = element_name + coords
                molecule.append(atom)
        print('Read and transcribed pdb for {}'.format(molecule_name))
        return molecule_name, molecule


@timer_logger
def config_loader(config_name='default_config'):
    """Sets up the desired global parameters from the config_file input.
    Allows different config settings for different projects, simply change the input config_name."""

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
    """Appends a message to the log file in a specific format."""

    with open(log_file, 'a+') as file:
        file.write(f'~~~~~~~~{message.upper()}~~~~~~~~\n\n-------------------------------------------------------\n\n')


def get_overage(molecule):
    """Bodge."""

    overage_dict = {'methane': 12.0, 'ethane': 16.0, 'acetone': 20.0, 'benzene': 24.0, 'methanol': 17.0}
    return overage_dict[molecule]


def pretty_progress():

    # Find the path of all files starting with QUBEKit_log and add their full path to log_files list
    log_files = []
    for root, dirs, files in walk('.', topdown=True):
        if 'QUBEKit' in root and '.tmp' not in root:
            log_files.append(f'{root}/QUBEKit_log_{root[10:]}')

    # Open all log files sequentially
    info = OrderedDict()
    for file in log_files:
        name = file.split('_')[1]

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
                # \033[ is the exit code; 1;32 are the style (bold) and colour (green); m reenters the code block.
                # The second exit code resets the style back to default.
                print('\033[1;32m{:>12d}\033[0;0m'.format(info[key_a][key_b]), end=' ')
            else:
                print('\033[1;31m{:>12d}\033[0;0m'.format(info[key_a][key_b]), end=' ')
        # TODO Print tick mark after final column (use unicode characters)
        # TODO Change red to error, orange to not done yet
        # Print statements end with \n by default so adding a space adds a newline. Weird.
        # Could also replace with print('\n', end=' ') which may be clearer ... ?
        print('')

    return


def set_dict_val(file_name, search_term):

    with open(file_name, 'r+') as file:
        for line in file:
            if search_term in line:
                return 1
    return 0


def unpickle(pickle_jar):
    """Function to unpickle a set of ligand objects from the pickle file, and return a dictionary of liagnds
    indexed by their progress."""

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
