#!/usr/bin/env python


from QUBEKit.smiles import smiles_to_pdb, smiles_mm_optimise
from QUBEKit.modseminario import ModSeminario
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.engines import PSI4, Chargemol, Gaussian
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan
from QUBEKit.parametrisation import OpenFF, AnteChamber, XML
from QUBEKit.helpers import config_loader, get_mol_data_from_csv, generate_config_csv, append_to_log, pretty_progress, pretty_print
from QUBEKit.decorators import exception_logger_decorator

from sys import argv as cmdline
from os import mkdir, chdir, path, rename
from shutil import copy


class Main:
    """Interprets commands from the terminal.
    Stores defaults or executes relevant functions.
    Will also create log and working directory where needed.
    See README.md for detailed discussion of QUBEKit commands.
    """

    def __init__(self):

        # Configs:
        self.defaults_dict = get_mol_data_from_csv()['default']
        self.qm, self.fitting, self.descriptions = config_loader(self.defaults_dict['config'])

        self.file, self.commands = self.parse_commands()
        self.create_log()
        self.execute()
        self.log_file = None

    start_up_msg = (f'If QUBEKit ever breaks or you would like to view timings and loads of other info, view the log file\n'
                    'Our documentation (README.md) also contains help on handling the various commands for QUBEKit')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__!r})'

    def parse_commands(self):
        """Parses commands from the terminal.
        Will either just return the commands to be used by execute();
        Update the configs with the commands given;
        Generate a .csv with the headers and defaults for bulk analysis;
        Recursively perform bulk analysis for smiles or pdbs;
        Perform a single analysis using a pdb or smiles string;
        Scan all log files and print matrix showing their progress (-progress command).
        """

        # Check for config changes or csv generation first.
        # Multiple configs can be changed at once
        self.commands = cmdline[1:]
        print('\nThese are the commands you gave:', self.commands)

        for count, cmd in enumerate(self.commands):

            # Change defaults for each analysis.
            if cmd == '-csv':
                csv_name = self.commands[count + 1]
                generate_config_csv(csv_name)
                # Move file to config folder.
                rename(csv_name, f'configs/{csv_name}')
                exit()

            if cmd == '-c':
                self.defaults_dict['charge'] = int(self.commands[count + 1])

            if cmd == '-m':
                self.defaults_dict['multiplicity'] = int(self.commands[count + 1])

            if cmd == '-ddec':
                self.qm['ddec version'] = int(self.commands[count + 1])

            if any(s in cmd for s in ('-geo', '-geometric')):
                self.qm['geometric'] = False if self.commands[count + 1] == 'false' else True

            if cmd == '-bonds':
                self.qm['bonds engine'] = str(self.commands[count + 1])

            if any(s in cmd for s in ('-cmol', '-chargemol')):
                self.qm['charges engine'] = 'chargemol'

            if cmd == '-onetep':
                self.qm['charges engine'] = 'onetep'

            if cmd == '-log':
                self.descriptions['log'] = str(self.commands[count + 1])

            if cmd == '-config':
                self.defaults_dict['config'] = str(self.commands[count + 1])

            if cmd == '-solvent':
                self.qm['solvent'] = False if self.commands[count + 1] == 'false' else True

            if cmd == '-param':
                self.fitting['parameter engine'] = str(self.commands[count + 1])

            if cmd == '-progress':
                pretty_progress()
                exit()

        print('These are the current defaults:', self.defaults_dict, '\nPlease note, some values may not be used.')

        # Then check what kind of analysis is being done.
        for count, cmd in enumerate(self.commands):

            # Controls high throughput.
            # Basically just runs the same functions but forces certain defaults.
            # '-bulk pdb example.csv' searches for local pdbs, runs analysis for each, defaults are from the csv file.
            # '-bulk smiles example.csv' will run analysis for all smile strings in the example.csv file.
            if cmd == '-bulk':

                csv_file = self.commands[count + 2]
                print(self.start_up_msg)

                if self.commands[count + 1] == 'smiles' or self.commands[count + 1] == 'pdb':

                    bulk_data = get_mol_data_from_csv(csv_file)

                    # Run full analysis for each smiles string or pdb in the .csv file
                    names = list(bulk_data.keys())[1:]
                    for name in names:

                        print(f'Currently analysing: {name}')
                        if self.commands[count + 1] == 'smiles':
                            smile_string = bulk_data[name]['smiles string']
                            self.file = smiles_to_pdb(smile_string, name)
                        elif self.commands[count + 1] == 'pdb':
                            self.file = name + '.pdb'
                        else:
                            raise Exception('The -bulk command is for smiles strings or pdb files only. The commands are: smiles, pdb')
                        self.defaults_dict = get_mol_data_from_csv(csv_file)[name]
                        self.create_log()
                        self.execute()
                        chdir('../../QUBEKit')
                    print('Finished bulk run. Use the command -progress to view which stages are complete.')
                    exit()

                else:
                    raise Exception('Bulk commands only supported for pdb files or csv file containing smiles strings. '
                                    'Please specify the type of bulk analysis you are doing, '
                                    'and include the name of the csv file defaults are to be extracted from.')

            else:
                # Check if a smiles string is given. If it is, generate the pdb and optimise it.
                if any(s in cmd for s in ('-sm', '-smiles')):

                    # Generate pdb from smiles string.
                    self.file = smiles_to_pdb(self.commands[count + 1])
                    self.defaults_dict['smiles string'] = self.commands[count + 1]
                    print(self.start_up_msg)
                    return self.file, self.commands

                # If a pdb is given instead, use that.
                elif 'pdb' in cmd:

                    self.file = cmd
                    print(self.start_up_msg)
                    return self.file, self.commands

                elif '-pickle' in cmd:
                    pickle_point = self.commands[count + 1]
                    # TODO Do stuff
                    pass

                # If neither a smiles string nor a pdb is given, raise exception.
                else:
                    raise Exception('Missing valid file type or smiles command. '
                                    'Please use pdb files and be sure to give the extension when typing the file name'
                                    ' into the terminal. '
                                    'Alternatively, use the smiles command (-sm) to generate a molecule.')

    def create_log(self):
        """Creates the working directory for the job as well as the log file.
        This log file is then extended every time the timer_logger decorator is called.
        """

        # Make working directory with correct name and move into it.
        from datetime import datetime

        date = datetime.now().strftime('%Y_%m_%d')

        # Define name of working directory.
        # This is formatted as 'QUBEKit_molecule name_yyyy_mm_dd_log_string'.
        log_string = f'QUBEKit_{self.file[:-4]}_{date}_{self.descriptions["log"]}'
        mkdir(log_string)

        # Copy active pdb into new directory.
        abspath = path.abspath(self.file)
        copy(abspath, f'{log_string}/{self.file}')
        # Move into new working directory.
        chdir(log_string)

        # Create log file in working directory.
        # This is formatted as 'QUBEKit_log_molecule name_yyyy_mm_dd_log_string'.

        self.log_file = f'QUBEKit_log_{self.file[:-4]}_{date}_{self.descriptions["log"]}'

        with open(self.log_file, 'w+') as log_file:

            log_file.write(f'Beginning log file: {datetime.now()}\n\n')
            log_file.write(f'The commands given were: {self.commands}\n\n')
            log_file.write(f'Analysing: {self.file[:-4]}\n\n')
            log_file.write('The defaults used are:\n')

            # Writes the config dictionaries to the log file.
            dicts_to_print = [self.defaults_dict, self.qm, self.fitting, self.descriptions]
            for dic in dicts_to_print:
                for key, var in dic.items():
                    log_file.write(f'{key}: {var}\n')
                log_file.write('\n')

            log_file.write('\n')

        return

    @exception_logger_decorator
    def execute(self):
        """Calls all the relevant classes and methods for the full QM calculation in the correct order.
        The log files associated with the particular run are changed via this method too.
        Exceptions are added to log (if raised) and key successful stages are added in caps.
        These stages can then be tracked with the -progress command (helpers.pretty_progress() function).
        """

        # TODO Add proper entry and exit points with pickle to allow more customised analysis.

        # Initialise file with pdb params from smiles string input or pdb input.
        mol = Ligand(self.file)

        # Having been provided a smiles string or pdb, perform a preliminary optimisation.

        # Optimise the molecule from the pdb.
        mol.filename, mol.descriptors = smiles_mm_optimise(mol.filename)

        # Initialise the molecule's pdb with its optimised form.
        mol.read_pdb(MM=True)

        # Parametrisation options:
        param_dict = {'openff': OpenFF, 'antechamber': AnteChamber, 'xml': XML}
        param_dict[self.fitting['parameter engine']](mol)

        append_to_log(self.log_file, f'Parametrised molecule with {self.fitting["parameter engine"]}')
        mol.pickle(state='parametrised')

        # Bonds engine options
        engine_dict = {'g09': Gaussian, 'psi4': PSI4}
        qm_engine = engine_dict[self.qm['bonds engine']](mol, self.defaults_dict)

        if self.qm['geometric']:

            # Calc geometric-related gradient and geometry
            qm_engine.geo_gradient(MM=True)
            mol.read_xyz_geo()

        else:
            qm_engine.generate_input(MM=True, optimize=True)
            qm_engine.optimised_structure()

        # Pickle
        append_to_log(self.log_file, f'Optimised structure calculated{" with geometric" if self.qm["geometric"] else ""}')
        mol.pickle(state='optimised')

        # Write input file for PSI4
        qm_engine.generate_input(QM=True, hessian=True)

        # Calc bond lengths from molecule topology
        mol.get_bond_lengths(QM=True)

        # Extract Hessian
        mol = qm_engine.hessian()

        # Modified Seminario for bonds and angles
        mod_sem = ModSeminario(mol, self.defaults_dict)
        mod_sem.modified_seminario_method()
        mol = qm_engine.all_modes()

        # Pickle
        append_to_log(self.log_file, 'Modified Seminario method complete')
        mol.pickle(state='mod-sem')

        # Prepare for density calc
        g09 = Gaussian(mol, self.defaults_dict)
        g09.generate_input(QM=True, density=True, solvent=self.qm['solvent'])

        # Pickle
        append_to_log(self.log_file, 'Gaussian analysis complete')
        mol.pickle(state='gaussian')

        # Perform DDEC calc
        c_mol = Chargemol(mol, self.defaults_dict)
        c_mol.generate_input()

        # Pickle
        append_to_log(self.log_file, f'Chargemol analysis with DDEC{self.qm["ddec version"]} complete')
        mol.pickle(state='chargemol')

        # Calculate Lennard-Jones parameters
        lj = LennardJones(mol, self.defaults_dict)
        mol = lj.amend_sig_eps()

        # Pickle
        append_to_log(self.log_file, 'Lennard-Jones parameters calculated')
        mol.pickle(state='l-j')

        # Perform torsion scan
        # scan = TorsionScan(mol, qm_engine, 'OpenMM')
        # sub_call(f'{scan.cmd}', shell=True)
        # scan.start_scan()

        append_to_log(self.log_file, 'Torsion scans complete')
        mol.pickle(state='torsions')

        mol.write_parameters()

        # Print ligand objects to terminal and log file
        pretty_print(mol, to_file=True)
        pretty_print(mol)

        return


if __name__ == '__main__':

    Main()
