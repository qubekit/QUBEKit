#!/usr/bin/env python3

from QUBEKit.engines.base_engine import Engines
from QUBEKit.utils import constants
from QUBEKit.utils.decorators import for_all_methods, timer_logger
from QUBEKit.utils.helpers import append_to_log, check_symmetry

import subprocess as sp

import numpy as np


@for_all_methods(timer_logger)
class PSI4(Engines):
    """
    Writes and executes input files for psi4.
    Also used to extract Hessian matrices; optimised structures; frequencies; etc.
    """

    def __init__(self, molecule):

        super().__init__(molecule)

        self.functional_dict = {'pbepbe': 'PBE', 'wb97xd': 'wB97X-D'}
        # Search for functional in dict, if it's not there, just leave the theory as it is.
        self.molecule.theory = self.functional_dict.get(self.molecule.theory.lower(), self.molecule.theory)

        # Test if PSI4 is callable
        psi4_test = sp.Popen('psi4 -h', shell=True, stdout=sp.PIPE)
        output = psi4_test.communicate()[0].decode('utf-8')
        if not output.startswith('usage:'):
            raise ModuleNotFoundError(
                'PSI4 not working. Please ensure PSI4 is installed and can be called with the command: psi4')

        if self.molecule.geometric:
            geo_test = sp.Popen('geometric-optimize -h', shell=True, stdout=sp.PIPE)
            output = geo_test.communicate()[0].decode('utf-8')
            if not output.startswith('usage: '):
                raise ModuleNotFoundError(
                    'Geometric not working. Please ensure geometric is installed and can be called '
                    'with the command: geometric-optimize')

    # TODO add restart from log method
    def generate_input(self, input_type='input', optimise=False, hessian=False, density=False, energy=False,
                       fchk=False, restart=False, execute=True):
        """
        Converts to psi4 input format to be run in psi4 without using geometric.
        :param input_type: The coordinate set of the molecule to be used
        :param optimise: Optimise the molecule to the desired convergence critera with in the iteration limit
        :param hessian: Calculate the hessian matrix
        :param density: Calculate the electron density
        :param energy: Calculate the single point energy of the molecule
        :param fchk: Write out a gaussian style Fchk file
        :param restart: Restart the calculation from a log point
        :param execute: Run the desired Psi4 job
        :return: The completion status of the job True if successful False if not run or failed
        """

        setters = ''
        tasks = ''

        if energy:
            append_to_log('Writing psi4 energy calculation input')
            tasks += f"\nenergy('{self.molecule.theory}')"

        if optimise:
            append_to_log('Writing PSI4 optimisation input', 'minor')
            setters += f' g_convergence {self.molecule.convergence}\n GEOM_MAXITER {self.molecule.iterations}\n'
            tasks += f"\noptimize('{self.molecule.theory.lower()}')"

        if hessian:
            append_to_log('Writing PSI4 Hessian matrix calculation input', 'minor')
            setters += ' hessian_write on\n'

            tasks += f"\nenergy, wfn = frequency('{self.molecule.theory.lower()}', return_wfn=True)"

            tasks += '\nwfn.hessian().print_out()\n\n'

        if density:
            pass
        #     append_to_log('Writing PSI4 density calculation input', 'minor')
        #     setters += " cubeprop_tasks ['density']\n"
        #
        #     overage = get_overage(self.molecule.name)
        #     setters += ' CUBIC_GRID_OVERAGE [{0}, {0}, {0}]\n'.format(overage)
        #     setters += ' CUBIC_GRID_SPACING [0.13, 0.13, 0.13]\n'
        #     tasks += f"grad, wfn = gradient('{self.molecule.theory.lower()}', return_wfn=True)\ncubeprop(wfn)"

        if fchk:
            append_to_log('Writing PSI4 input file to generate fchk file')
            tasks += f"\ngrad, wfn = gradient('{self.molecule.theory.lower()}', return_wfn=True)"
            tasks += '\nfchk_writer = psi4.core.FCHKWriter(wfn)'
            tasks += f'\nfchk_writer.write("{self.molecule.name}_psi4.fchk")\n'

        # TODO If overage cannot be made to work, delete and just use Gaussian.
        # if self.molecule.solvent:
        #     setters += ' pcm true\n pcm_scf_type total\n'
        #     tasks += '\n\npcm = {'
        #     tasks += '\n units = Angstrom\n Medium {\n  SolverType = IEFPCM\n  Solvent = Chloroform\n }'
        #     tasks += '\n Cavity {\n  RadiiSet = UFF\n  Type = GePol\n  Scaling = False\n  Area = 0.3\n  Mode = Implicit'
        #     tasks += '\n }\n}'

        setters += '}\n'

        if not execute:
            setters += f'set_num_threads({self.molecule.threads})\n'

        # input.dat is the PSI4 input file.
        with open('input.dat', 'w+') as input_file:
            # opening tag is always writen
            input_file.write(f'memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n'
                             f'{self.molecule.charge} {self.molecule.multiplicity} \n')
            # molecule is always printed
            for i, atom in enumerate(self.molecule.coords[input_type]):
                input_file.write(f' {self.molecule.atoms[i].element}    '
                                 f'{float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f} \n')

            input_file.write(f" units angstrom\n no_reorient\n}}\n\nset {{\n basis {self.molecule.basis}\n")

            input_file.write(setters)
            input_file.write(tasks)

        if execute:
            with open('log.txt', 'w+') as log:
                sp.run(f'psi4 input.dat -n {self.molecule.threads}', shell=True, stdout=log, stderr=log)

            # Now check the exit status of the job
            return self.check_for_errors()

        else:
            return {'success': False, 'error': 'Not run'}

    def check_for_errors(self):
        """
        Read the output file from the job and check for normal termination and any errors
        :return: A dictionary of the success status and any problems.
        """

        with open('output.dat', 'r') as log:
            for line in log:
                if '*** Psi4 exiting successfully.' in line:
                    return {'success': True}

                elif '*** Psi4 encountered an error.' in line:
                    return {'success': False,
                            'error': 'Not known'}

            return {'success': False,
                    'error': 'Segfault'}

    def hessian(self):
        """
        Parses the Hessian from the output.dat file (from psi4) into a numpy array;
        performs check to ensure it is symmetric;
        has some basic error handling for if the file is missing data etc.
        """

        hess_size = 3 * len(self.molecule.atoms)

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:

            lines = file.readlines()

            for count, line in enumerate(lines):
                if '## Hessian' in line or '## New Matrix (Symmetry' in line:
                    # Set the start of the hessian to the row of the first value.
                    hess_start = count + 5
                    break
            else:
                raise EOFError('Cannot locate Hessian matrix in output.dat file.')

            # Check if the hessian continues over onto more lines (i.e. if hess_size is not divisible by 5)
            extra = 0 if hess_size % 5 == 0 else 1

            # hess_length: # of cols * length of each col
            #            + # of cols - 1 * #blank lines per row of hess_vals
            #            + # blank lines per row of hess_vals if the hess_size continues over onto more lines.
            hess_length = (hess_size // 5) * hess_size + (hess_size // 5 - 1) * 3 + extra * (3 + hess_size)

            hess_end = hess_start + hess_length

            hess_vals = []

            for file_line in lines[hess_start:hess_end]:
                # Compile lists of the 5 Hessian floats for each row.
                # Number of floats in last row may be less than 5.
                # Only the actual floats are added, not the separating numbers.
                row_vals = [float(val) for val in file_line.split() if len(val) > 5]
                hess_vals.append(row_vals)

            # Remove blank list entries
            hess_vals = [elem for elem in hess_vals if elem]

            reshaped = []

            # Convert from list of (lists, length 5) to 2d array of size hess_size x hess_size
            for old_row in range(hess_size):
                new_row = []
                for col_block in range(hess_size // 5 + extra):
                    new_row += hess_vals[old_row + col_block * hess_size]

                reshaped.append(new_row)

            hess_matrix = np.array(reshaped)

            # Cache the unit conversion.
            conversion = constants.HA_TO_KCAL_P_MOL / (constants.BOHR_TO_ANGS ** 2)
            # Element-wise multiplication
            hess_matrix *= conversion

            check_symmetry(hess_matrix)

            return hess_matrix

    def optimised_structure(self):
        """
        Parses the final optimised structure from the output.dat file (from psi4) to a numpy array.
        Also returns the energy of the optimized structure.
        """

        # Run through the file and find all lines containing '==> Geometry', add these lines to a list.
        # Reverse the list
        # from the start of this list, jump down to the first atom and set this as the start point
        # Split the row into 4 columns: centre, x, y, z.
        # Add each row to a matrix.
        # Return the matrix.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            # Will contain index of all the lines containing '==> Geometry'.
            geo_pos_list = []
            for count, line in enumerate(lines):
                if '==> Geometry' in line:
                    geo_pos_list.append(count)

                elif '**** Optimization is complete!' in line:
                    opt_pos = count
                    opt_steps = int(line.split()[5])

            if not (opt_pos and opt_steps):
                raise EOFError('According to the output.dat file, optimisation has not completed.')

            # now get the final opt_energy
            opt_energy = float(lines[opt_pos + opt_steps + 7].split()[1])

            # Set the start as the last instance of '==> Geometry'.
            start_of_vals = geo_pos_list[-1] + 9

            opt_struct = []

            for row in range(len(self.molecule.atoms)):

                # Append the first 4 columns of each row, converting to float as necessary.
                struct_row = []
                for indx in range(3):
                    struct_row.append(float(lines[start_of_vals + row].split()[indx + 1]))

                opt_struct.append(struct_row)

        return np.array(opt_struct), opt_energy

    @staticmethod
    def get_energy():
        """Get the energy of a single point calculation."""

        # open the psi4 log file
        with open('output.dat', 'r') as log:
            for line in log:
                if 'Total Energy =' in line:
                    return float(line.split()[3])

        raise EOFError('Cannot find energy in output.dat file.')

    def all_modes(self):
        """Extract all modes from the psi4 output file."""

        # Find "post-proj  all modes"
        # Jump to first value, ignoring text.
        # Move through data, adding it to a list
        # continue onto next line.
        # Repeat until the following line is known to be empty.

        # output.dat is the psi4 output file.
        with open('output.dat', 'r') as file:
            lines = file.readlines()
            for count, line in enumerate(lines):
                if "post-proj  all modes" in line:
                    start_of_vals = count
                    break
            else:
                raise EOFError('Cannot locate modes in output.dat file.')

            # Barring the first (and sometimes last) line, dat file has 6 values per row.
            end_of_vals = start_of_vals + (3 * len(self.molecule.atoms)) // 6

            structures = lines[start_of_vals][24:].replace("'", "").split()
            structures = structures[6:]

            for row in range(1, end_of_vals - start_of_vals):
                # Remove double strings and weird formatting.
                structures += lines[start_of_vals + row].replace("'", "").replace("]", "").split()

            all_modes = [float(val) for val in structures]

            return np.array(all_modes)

    def geo_gradient(self, input_type='input', threads=False, execute=True):
        """
        Write the psi4 style input file to get the gradient for geometric
        and run geometric optimisation.
        """

        molecule = self.molecule.coords[input_type]

        with open(f'{self.molecule.name}.psi4in', 'w+') as file:

            file.write(f'memory {self.molecule.memory} GB\n\nmolecule {self.molecule.name} {{\n {self.molecule.charge} {self.molecule.multiplicity} \n')

            for i, atom in enumerate(molecule):
                file.write(f'  {self.molecule.atoms[i].element:2}    {float(atom[0]): .10f}  {float(atom[1]): .10f}  {float(atom[2]): .10f}\n')

            file.write(f' units angstrom\n no_reorient\n}}\nset basis {self.molecule.basis}\n')

            if threads:
                file.write(f'set_num_threads({self.molecule.threads})')

            file.write(f"\n\ngradient('{self.molecule.theory}')\n")

        if execute:
            with open('log.txt', 'w+') as log:
                sp.run(f'geometric-optimize --psi4 {self.molecule.name}.psi4in {self.molecule.constraints_file} '
                       f'--nt {self.molecule.threads}', shell=True, stdout=log, stderr=log)
