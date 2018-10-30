#!/usr/bin/env python


from sys import argv as cmdline
from os import mkdir, chdir
from subprocess import call as sub_call

from QUBEKit.engines import PSI4, Geometric


defaults_dict = {'charge': 0, 'multiplicity': 1,
                 'bonds engine': 'psi4', 'charges engine': 'chargemol',
                 'ddec version': 6, 'geometric': True, 'solvent': True,
                 'run number': '999', 'config': 'default_config'}

# TODO Complete help dict to cover everything.
help_dict = {'bonds': 'Bonds help, default={}'.format(defaults_dict['bonds engine']),
             'charges': 'Charges help, default={}'.format(defaults_dict['charges engine']),
             'dihedrals': 'Dihedrals help, default=???',
             'smiles': 'Enter the smiles code for the molecule.',
             '-c': 'The charge of the molecule, default={}'.format(defaults_dict['charge']),
             '-m': 'The multiplicity of the molecule, default={}'.format(defaults_dict['multiplicity']),
             'qubekit': 'Welcome to QUBEKit\n{}'.format(__doc__),
             'ddec': 'It is recommended to use the default DDEC6, however DDEC3 is available.',
             'log': 'The name applied to the log files and directory for the current analysis.'}


def main():
    """Interprets commands from the terminal. Stores defaults, returns help or executes relevant functions."""

    commands = cmdline[1:]
    print('These are the commands you gave:', commands)

    for count, cmd in enumerate(commands):

        if any(s in cmd for s in ('-h', '-help')):

            if commands[count + 1] in help_dict:
                print(help_dict[commands[count + 1]])
            else:
                print('Unrecognised help command; the valid help commands are:', help_dict.keys())
            return

        if any(s in cmd for s in ('-sm', '-smiles')):

            import smiles

            smiles.smiles_to_pdb(commands[count + 1])
            return

        # Changing defaults for each test run
        if cmd == '-c':
            defaults_dict['charge'] = int(commands[count + 1])

        if cmd == '-m':
            defaults_dict['multiplicity'] = int(commands[count + 1])

        if cmd == '-ddec':
            defaults_dict['ddec version'] = int(commands[count + 1])

        if any(s in cmd for s in ('geo', 'geometric')):
            defaults_dict['geometric'] = False if commands[count + 1] == 'False' or 'false' else True

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
            defaults_dict['solvent'] = False if commands[count + 1] == 'False' or 'false' else True

    print('These are the current defaults:', defaults_dict, 'Please note, some values may not be used.')

    for count, cmd in enumerate(commands):

        if 'pdb' in cmd:

            file = cmd
            # Make working directory here with correct name.
            log_string = file[:-4] + '_' + defaults_dict['run number']
            mkdir(log_string)
            # Copy active pdb into new directory.
            sub_call(f'cp {file} {str(log_string) + "/" + file}', shell=True)
            # Change into new directory.
            chdir(log_string)

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

                    # Initialisations
                    call_psi4 = PSI4(file, defaults_dict['config'], defaults_dict['geometric'], defaults_dict['solvent'])
                    call_geo = Geometric(file, defaults_dict['config'])

                    # Calls
                    call_psi4.generate_input(defaults_dict['charge'], defaults_dict['multiplicity'])
                    if defaults_dict['geometric']:
                        call_geo.pdb_to_psi4_geo(defaults_dict['charge'], defaults_dict['multiplicity'])
                        opt_mol = call_geo.optimised_structure()
                        call_psi4.generate_input(defaults_dict['charge'], defaults_dict['multiplicity'])
                    return

                elif defaults_dict['bonds engine'] == 'g09':
                    # TODO Add g09 engine function
                    return

                else:
                    raise Exception('Invalid bond engine given. Please use either psi4 or g09.')
        else:
            raise Exception('''Missing valid file type or smiles command.
            Please use zmat or pdb files and be sure to give the extension when typing the file name into the terminal.
            Alternatively, use the smiles command to generate a molecule.''')


if __name__ == '__main__':

    main()