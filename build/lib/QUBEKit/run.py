#!/usr/bin/env python


from QUBEKit import smiles
from QUBEKit.modseminario import ModSeminario
from QUBEKit.lennard_jones import LennardJones
from QUBEKit.engines import PSI4, Chargemol, Gaussian
from QUBEKit.ligand import Ligand
from QUBEKit.dihedrals import TorsionScan
from QUBEKit.parametrisation import OpenFF, AnteChamber
from QUBEKit.helpers import config_loader, get_mol_data_from_csv, generate_config_csv

from sys import argv as cmdline
from os import mkdir, chdir, listdir, path, rename


class Main:
    """Interprets commands from the terminal.
    Stores defaults or executes relevant functions.
    Will also create log and working directory where needed.
    """

    def __init__(self):

        # Configs:
        self.defaults_dict = get_mol_data_from_csv()['default']
        self.qm, self.fitting, self.descriptions = config_loader(self.defaults_dict['config'])

        self.file, self.commands = self.parse_commands()
        self.create_log()
        self.execute()

    def parse_commands(self):
        """Parses commands from the terminal.
        Will either just return the commands to be used by execute();
        Update the configs with the commands given;
        Recursively perform bulk analysis for smiles or pdbs.
        """

        # Parse the terminal inputs
        self.commands = cmdline[1:]
        print('These are the commands you gave:', self.commands)

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

            if any(s in cmd for s in ('geo', 'geometric')):
                self.qm['geometric'] = False if self.commands[count + 1] == 'false' else True

            if cmd == 'psi4':
                self.qm['bonds engine'] = 'psi4'

            if cmd == 'g09':
                self.qm['bonds engine'] = 'g09'

            if any(s in cmd for s in ('cmol', 'chargemol')):
                self.qm['charges engine'] = 'chargemol'

            if cmd == 'onetep':
                self.qm['charges engine'] = 'onetep'

            if cmd == '-log':
                self.descriptions['log'] = str(self.commands[count + 1])

            if cmd == '-config':
                self.defaults_dict['config'] = str(self.commands[count + 1])

            if cmd == '-solvent':
                self.qm['solvent'] = False if self.commands[count + 1] == 'false' else True

            if cmd == '-param':
                self.fitting['parameter_engine'] = str(self.commands[count + 1])

        print('These are the current defaults:', self.defaults_dict, '\nPlease note, some values may not be used.')

        for count, cmd in enumerate(self.commands):

            # Controls high throughput.
            # Basically just runs the same functions but forces certain defaults.
            # '-bulk pdb example.csv' searches for local pdbs, runs analysis for each, defaults are from the csv file.
            # '-bulk smile example.csv' will run analysis for all smile strings in the example.csv file.
            if cmd == '-bulk':

                csv_file = self.commands[count + 2]

                if self.commands[count + 1] == 'smile':

                    smile_data = get_mol_data_from_csv(csv_file)

                    # Run full analysis for each smile string in the .csv file
                    names = list(smile_data.keys())[1:]
                    for name in names:

                        smile_string = smile_data[name]['smile string']
                        print(f'Currently analysing: {name}')
                        self.file = smiles.smiles_to_pdb(smile_string, name)
                        self.defaults_dict = get_mol_data_from_csv(csv_file)[name]
                        self.create_log()
                        self.execute()
                        # For some bizarre (and infuriating) reason, sub_call('cd ../', shell=True) doesn't work
                        chdir('../../QUBEKit')
                    exit()

                elif self.commands[count + 1] == 'pdb':

                    # Find all pdb files in current directory.
                    files = [file for file in listdir('.') if path.isfile(file)]
                    pdbs = [file for file in files if file.endswith('.pdb')]

                    # Run full analysis for each pdb provided. Pdb names are used as the keys for the csv reader.
                    for pdb in pdbs:
                        print(f'Currently analysing: {pdb[:-4]}')
                        self.file = pdb
                        self.defaults_dict = get_mol_data_from_csv(csv_file)[pdb[:-4]]
                        self.create_log()
                        self.execute()
                        chdir('../../QUBEKit')
                    exit()

                else:
                    raise Exception('Bulk commands only supported for pdb files or a file of smiles strings.'
                                    'Please specify the type of bulk analysis you are doing, '
                                    'and include the name of the csv file defaults are extracted from.')

            else:
                # Check if a smiles string is given. If it is, generate the pdb and optimise it.
                if any(s in cmd for s in ('-sm', '-smiles')):

                    # Generate pdb from smiles string.
                    self.file = smiles.smiles_to_pdb(self.commands[count + 1])
                    return self.file, self.commands

                # If a pdb is given instead, use that.
                elif 'pdb' in cmd:

                    self.file = cmd
                    return self.file, self.commands

                # If neither a smiles string nor a pdb is given, raise exception.
                else:
                    raise Exception('''Missing valid file type or smiles command.
                        Please use pdb files and be sure to give the extension when typing the file name into the terminal.
                        Alternatively, use the smiles command (-sm) to generate a molecule.''')

    def create_log(self):
        """Creates the working directory for the job as well as the log file.
        This log file is then extended every time the timer_logger decorator is called.
        """

        # Make working directory with correct name and move into it.
        from datetime import datetime

        date = datetime.now().strftime('%Y_%m_%d')

        # Define name of working directory.
        # This is formatted as 'QUBEKit_molecule name_yy_mm_dd_log_string'.
        log_string = f'QUBEKit_{self.file[:-4]}_{date}_{self.descriptions["log"]}'
        mkdir(log_string)

        # Copy active pdb into new directory.
        rename(self.file, f'{log_string}/{self.file}')
        # Move into new working directory.
        chdir(log_string)

        # Create log file.
        # This is formatted as 'QUBEKit_log_molecule name_yy_mm_dd_log_string'.
        with open(f'QUBEKit_log_{self.file[:-4]}_{date}_{self.descriptions["log"]}', 'w+') as log_file:

            log_file.write(f'Beginning log file: {datetime.now()}\n\n')
            log_file.write(f'The commands given were: {self.commands}\n\n')
            log_file.write('The defaults used are:\n')

            # Writes the config dictionaries to the log file.
            dicts_to_print = [self.defaults_dict, self.qm, self.fitting, self.descriptions]
            for dic in dicts_to_print:
                for key, var in dic.items():
                    log_file.write(f'{key}: {var}\n')
                log_file.write('\n')

            log_file.write('\n')

        print(f'If QUBEKit ever breaks or you would just like to view timings and other info, check the log file: \n'
              f'QUBEKit_log_{self.file[:-4]}_{date}_{self.descriptions["log"]}\n'
              'Our documentation (README.md) also contains help on handling the various commands for QUBEKit')

        return

    def execute(self):
        """Calls all the relevant classes and methods for the full QM calculation in the correct order.
        # TODO Add proper entry and exit points to allow more customised analysis.
        """

        # Initialise file with pdb params from smiles string input or pdb input.
        mol = Ligand(self.file)

        # Having been provided a smiles string or pdb, perform a preliminary optimisation.

        # Optimise the molecule from the pdb.
        mol.filename, mol.descriptors = smiles.smiles_mm_optimise(mol.filename)

        # Initialise the molecule's pdb with its optimised form.
        mol.read_pdb(MM=True)

        # Parametrise with openFF

        if self.fitting['parameter engine'] == 'openff':
            OpenFF(mol)
        elif self.fitting['parameter engine'] == 'antechamber':
            AnteChamber(mol)
        # TODO Add others
        elif self.fitting['parameter engine'] == '':
            pass
        else:
            raise Exception('Invalid parametrisation engine, please select from openff, antechamber or XXXXXXX')

        if self.qm['bonds engine'] == 'psi4':
            # Initialise for PSI4
            qm_engine = PSI4(mol, self.defaults_dict)
        else:
            # qm_engine = OTHER
            # TODO Add one
            raise Exception('No other bonds engine currently implemented.')

        if self.qm['geometric']:

            # Calc geometric-related gradient and geometry
            qm_engine.geo_gradient(MM=True)
            mol.read_xyz_geo()

        else:
            qm_engine.generate_input(MM=True, optimize=True)
            qm_engine.optimised_structure()

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

        # # Prepare for density calc
        # g09 = Gaussian(mol, self.defaults_dict)
        # g09.generate_input(QM=True, density=True)
        #
        # # Perform DDEC calc
        # c_mol = Chargemol(mol, self.defaults_dict)
        # c_mol.generate_input()
        #
        # # Calculate Lennard-Jones parameters
        # lj = LennardJones(mol, self.defaults_dict)
        # mol = lj.amend_sig_eps()
        #
        # # Perform torsion scan
        # scan = TorsionScan(mol, qm_engine, 'OpenMM')
        # sub_call(f'{scan.cmd}', shell=True)
        # scan.start_scan()
        #
        # print(mol)

        return


if __name__ == '__main__':

    Main()
