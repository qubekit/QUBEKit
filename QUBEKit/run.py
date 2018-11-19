#!/usr/bin/env python


"""
Given a list of commands, such as: "bonds", "-h", "psi4" some are taken as single word commands.
Others however, such as changing defaults: ("-c", "0") ("-m", "1") etc are taken as tuple commands.
All tuple commands are preceded by a "-", while single word commands are not.
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

Each time the program runs, a new working directory containing a log file will be created.
The name of the directory will contain the log number provided (or the default).
This log file will store which methods were called, how long they took, and any docstring for them (if it exists).
If using QUBEKit multiple times per day with the same molecule, it is therefore necessary to update the 'run number'.
Not updating the run number when analysing the same molecule on the same day will prevent the program from running.
Updating the run number can be done with the command:
python run.py -log #
where '#' is any number or string you like.
"""


from sys import argv as cmdline
from os import mkdir, chdir
from subprocess import call as sub_call


from engines import PSI4
from ligand import Ligand
import smiles


defaults_dict = {
    'charge': 0, 'multiplicity': 1,
    'bonds engine': 'psi4', 'charges engine': 'chargemol',
    'ddec version': 6, 'geometric': True, 'solvent': True,
    'run number': '999', 'config': 'default_config'
}

# TODO Complete help dict to cover everything.
help_dict = {
    'bonds': 'Bonds help, default={}'.format(defaults_dict['bonds engine']),
    'charges': 'Charges help, default={}'.format(defaults_dict['charges engine']),
    'dihedrals': 'Dihedrals help, default=???',
    'smiles': 'Enter the smiles code for the molecule.',
    '-c': 'The charge of the molecule, default={}'.format(defaults_dict['charge']),
    '-m': 'The multiplicity of the molecule, default={}'.format(defaults_dict['multiplicity']),
    'qubekit': 'Welcome to QUBEKit\n{}'.format(__doc__),
    'ddec': 'It is recommended to use the default DDEC6, however DDEC3 is available.',
    'log': 'The name applied to the log files and directory for the current analysis.'
}


def main():
    """Interprets commands from the terminal.
    Stores defaults, returns help or executes relevant functions.
    """

    # Parse the terminal inputs
    commands = cmdline[1:]
    print('These are the commands you gave:', commands)

    for count, cmd in enumerate(commands):

        # Call help then stop.
        if any(s in cmd for s in ('-h', '-help')):

            if commands[count + 1] in help_dict:
                print(help_dict[commands[count + 1]])
            else:
                print('Unrecognised help command; the valid help commands are:', help_dict.keys())
            return

        # Change defaults for each analysis.
        if cmd == '-c':
            defaults_dict['charge'] = int(commands[count + 1])

        if cmd == '-m':
            defaults_dict['multiplicity'] = int(commands[count + 1])

        if cmd == '-ddec':
            defaults_dict['ddec version'] = int(commands[count + 1])

        if any(s in cmd for s in ('geo', 'geometric')):
            defaults_dict['geometric'] = False if commands[count + 1] == 'false' else True

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

        if cmd == '-solvent':
            defaults_dict['solvent'] = False if commands[count + 1] == 'false' else True

    print('These are the current defaults:', defaults_dict, 'Please note, some values may not be used.')

    for count, cmd in enumerate(commands):

        # Check if a smiles string is given. If it is, generate the pdb and optimise it.
        if any(s in cmd for s in ('-sm', '-smiles')):

            # Generate pdb from smiles string.
            file = smiles.smiles_to_pdb(commands[count + 1])

        # If a pdb is given instead, use that.
        elif 'pdb' in cmd:

            file = cmd

        # If neither a smiles string nor a pdb is given, raise exception.
        else:
            raise Exception('''Missing valid file type or smiles command.
            Please use zmat or pdb files and be sure to give the extension when typing the file name into the terminal.
            Alternatively, use the smiles command to generate a molecule.''')

        # Initialise file with pdb params from smiles string input or pdb input.
        mol = Ligand(file)

        # Make working directory with correct name and move into it.
        from datetime import datetime

        date = datetime.now().strftime('%Y_%m_%d')

        # Define name of working directory.
        # This is formatted as 'QUBEKit_yy_mm_dd_moleculename_lognum'.
        log_string = f'QUBEKit_{date}_{file[:-4]}_{defaults_dict["run number"]}'
        mkdir(log_string)

        # Copy active pdb into new directory.
        sub_call(f'cp {file} {str(log_string)}/{file}', shell=True)
        # Move into new working directory.
        chdir(log_string)

        # Create log file.
        # This is formatted as 'QUBEKit_log_yy_mm_dd_moleculename_logname/num'.
        with open(f'QUBEKit_log_{date}_{mol.name}_{defaults_dict["run number"]}', 'w+') as log_file:
            log_file.write(f'Beginning log file: {datetime.now()}\n\n')

        # Having been provided a smiles string or pdb, perform a preliminary optimisation.

        # Optimise the molecule from the pdb.
        mol.filename = smiles.smiles_mm_optimise(mol.filename)
        # Initialise the molecule's pdb with its optimised form.
        mol.read_pdb(MM=True)

        if 'charges' in commands:

            if defaults_dict['charges engine'] == 'chargemol':
                print('Running chargemol charge analysis.')
                return

            elif defaults_dict['charges engine'] == 'onetep':
                # TODO Add ONETEP engine function
                print('Running onetep charge analysis.')
                return

            else:
                raise Exception('Invalid charge engine given. Please use either chargemol or onetep.')

        elif 'dihedrals' in commands:

            # TODO Add dihedral calcs
            print('Running dihedrals ...')
            return

        # else: run default bonds analysis
        else:

            if defaults_dict['bonds engine'] == 'psi4':
                print('Running psi4 bonds analysis.')
                QMEngine = PSI4(mol, defaults_dict['config'], defaults_dict['geometric'], defaults_dict['solvent'])
                QMEngine.generate_input(defaults_dict['charge'], defaults_dict['multiplicity'])
                return

            elif defaults_dict['bonds engine'] == 'g09':
                # TODO Add g09 engine function
                return

            else:
                raise Exception('Invalid bond engine given. Please use either psi4 or g09.')


if __name__ == '__main__':

    main()
